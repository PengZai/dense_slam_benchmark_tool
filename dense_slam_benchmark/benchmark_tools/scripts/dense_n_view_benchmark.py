import json
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dense_slam_benchmark.benchmark_tools import metrics
from dense_slam_benchmark.benchmark_tools.dataloader import Testdataset
from dense_slam_benchmark.benchmark_tools.external import init_model
from dense_slam_benchmark.benchmark_tools.postprocessing import simple_postprocess
from dense_slam_benchmark.benchmark_tools.utils import saveMetricsLogAndResults, updateConfig


IGNORE_KEYS = {
    "idx", "name", "camera_id", "camera_name", "dataset", "scene_name",
    "data_norm_type", "ts", "undistorted_raw_image",
}

PC_METRIC_KEYS = (
    "acc_mean", "acc_median", "acc_inlier_ratio",
    "comp_mean", "comp_median", "comp_inlier_ratio",
)


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def _frames_to_device(frames, device):
    for frame in frames:
        for view in frame:
            for name in view.keys():
                if name in IGNORE_KEYS:
                    continue
                view[name] = view[name].to(device, non_blocking=True)


def _build_result_dict(input_view, res, batch_idx, model_name, GT_pointcloud):
    ts = input_view["ts"][batch_idx]
    if hasattr(ts, "item"):
        ts = ts.item()
    return {
        "basic": {
            "sample_idx": input_view["idx"][batch_idx].item(),
            "ts": ts,
            "dataset_name": input_view["dataset"][batch_idx],
            "scene_name": input_view["scene_name"][batch_idx],
            "name": input_view["name"][batch_idx],
            "camera_name": input_view["camera_name"][batch_idx],
            "camera_id": input_view["camera_id"][batch_idx].item(),
            "pred_model": model_name,
        },
        "pred": {
            "depth": np.expand_dims(res["pred_depth"][batch_idx], axis=-1),
            "depth_mask": res["pred_depth_mask"][batch_idx],
            "depth_confidence": res.get(
                "pred_depth_confidence",
                res["pred_depth_mask"].astype(np.float32),
            )[batch_idx],
            "T_w_c": res["pred_T_w_c"][batch_idx],
            "runtime": res["runtime"],
            "intrinsics": res["pred_intrinsics"][batch_idx],
        },
        "GT": {
            "undistorted_raw_image": input_view["undistorted_raw_image"][batch_idx].numpy(),
            "input_depth": np.expand_dims(input_view["input_depth"][batch_idx].cpu().numpy().squeeze(), axis=-1),
            "input_depth_mask": input_view["input_depth_mask"][batch_idx].cpu().numpy().squeeze(),
            "GT_depth": np.expand_dims(input_view["GT_depth"][batch_idx].cpu().numpy().squeeze(), axis=-1),
            "GT_depth_mask": input_view["GT_depth_mask"][batch_idx].cpu().numpy().squeeze(),
            "GT_pointcloud": GT_pointcloud,
            "T_w_c": input_view["T_w_c"][batch_idx].cpu().numpy(),
            "intrinsics": input_view["intrinsics"][batch_idx].cpu().numpy(),
        },
    }


def _per_frame_depth_metrics(r):
    pred_m = r["pred"]["depth_mask"]
    in_m = r["GT"]["input_depth_mask"]
    gt_m = r["GT"]["GT_depth_mask"]
    pred_in = pred_m & in_m
    pred_gt = pred_m & gt_m
    in_gt = in_m & gt_m

    GT_d, in_d, pred_d = r["GT"]["GT_depth"], r["GT"]["input_depth"], r["pred"]["depth"]

    m = {
        "runtime": r["pred"]["runtime"],
        "postprocess_time": r["pred"]["postprocess_time"],
        "num_valid_pred": pred_m.sum(),
        "num_valid_input_depth": in_m.sum(),
        "num_valid_GT_depth": gt_m.sum(),
        "num_valid_pred_vs_input_depth": pred_in.sum(),
        "num_valid_pred_vs_GT_depth": pred_gt.sum(),
        "num_input_depth_vs_GT_depth": in_gt.sum(),
        "GT_depth_vs_input_depth_rel_inlier_ratio": metrics.rel_thresh_inliers(GT_d, in_d, mask=in_gt),
        "GT_depth_vs_input_depth_m_rel_ae": metrics.m_rel_ae(GT_d, in_d, mask=in_gt),
        "GT_depth_vs_input_depth_abs_inlier_ratio": metrics.abs_thresh_inliers(GT_d, in_d, mask=in_gt),
        "GT_depth_vs_input_depth_m_ae": metrics.m_ae(GT_d, in_d, mask=in_gt),
    }
    if pred_m.any():
        m.update({
            "input_depth_vs_pred_depth_rel_inlier_ratio": metrics.rel_thresh_inliers(in_d, pred_d, mask=pred_in),
            "input_depth_vs_pred_depth_m_rel_ae": metrics.m_rel_ae(in_d, pred_d, mask=pred_in),
            "input_depth_vs_pred_depth_m_ae": metrics.m_ae(in_d, pred_d, mask=pred_in),
            "GT_depth_vs_pred_depth_rel_inlier_ratio": metrics.rel_thresh_inliers(GT_d, pred_d, mask=pred_gt),
            "GT_depth_vs_pred_depth_m_rel_ae": metrics.m_rel_ae(GT_d, pred_d, mask=pred_gt),
            "GT_depth_vs_pred_depth_abs_inlier_ratio": metrics.abs_thresh_inliers(GT_d, pred_d, mask=pred_gt),
            "GT_depth_vs_pred_depth_m_ae": metrics.m_ae(GT_d, pred_d, mask=pred_gt),
        })
    return m


def _scene_pc_metrics(all_GT, all_input, all_pred):
    def _safe(target, source):
        if target.shape[0] == 0 or source.shape[0] == 0:
            return None
        return metrics.pointcloud_accuracy(target, source) + metrics.pointcloud_completion(target, source)

    out = {}
    for prefix, target, source in [
        ("GT_pointcloud_vs_input_pointcloud", all_GT, all_input),
        ("input_pointcloud_vs_pred_pointcloud", all_input, all_pred),
        ("GT_pointcloud_vs_pred_pointcloud", all_GT, all_pred),
    ]:
        vals = _safe(target, source)
        if vals is not None:
            out.update({f"{prefix}_{k}": v for k, v in zip(PC_METRIC_KEYS, vals)})
    return out


def _stack(chunks, dim=3):
    if not chunks:
        return np.empty((0, dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def dense_n_view_benchmark(config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(config["root_data_dir"], config["scene_name"], "scene.json"), "r", encoding="utf-8") as f:
        camera_configs = json.load(f)
    config = updateConfig(config, camera_configs)

    test_dataset = Testdataset(config)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1)

    model = init_model(config.model.model_str, config.model.model_config, torch_hub_force_reload=False)
    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.eval()

    # ---- Inference ----
    with torch.no_grad():
        result_list = []
        num_sub_scenes = len(test_dataloader)
        for sub_scene_idx, frames in enumerate(test_dataloader):
            num_frame = len(frames)
            batch_size = frames[0][0]["undistorted_image"].shape[0]
            sample_idx_in_batch = [frames[0][0]["idx"][b].item() for b in range(batch_size)]
            print(
                f"[{sub_scene_idx + 1}/{num_sub_scenes}] "
                f"batch_size={batch_size}, num_frame={num_frame}, "
                f"sample_idx={sample_idx_in_batch}"
            )

            _frames_to_device(frames, device)
            results = model(frames)

            for batch_idx in range(batch_size):
                sub_scene = []
                for frame_idx in range(num_frame):
                    iv = frames[frame_idx][0]
                    cam_dataset = test_dataset.camera_dataset_by_id[iv["camera_id"][batch_idx].item()]
                    GT_pc = cam_dataset.getGTPointCloud(iv["idx"][batch_idx].item())
                    sub_scene.append(
                        _build_result_dict(iv, results[frame_idx], batch_idx, model.name, GT_pc)
                    )
                result_list.append(sub_scene)

    result_list = simple_postprocess(config, result_list)

    # ---- Per-frame depth metrics + accumulate world-frame point clouds (single pass) ----
    pred_chunks, pred_color_chunks, input_chunks, GT_chunks = [], [], [], []
    for r in result_list:
        r["metrics"] = _per_frame_depth_metrics(r)

        cam_dataset = test_dataset.camera_dataset_by_id[r["basic"]["camera_id"]]
        sample_idx = r["basic"]["sample_idx"]
        input_pc = cam_dataset.getInputPointCloud(sample_idx)
        GT_pc = r["GT"]["GT_pointcloud"]

        pred_mask = r["pred"]["depth_mask"]
        scale = r["pred"].get("similarity_scale", 1.0)
        pred_pts = r["pred"]["pts3d"][pred_mask] * scale
        pred_T = np.asarray(r["pred"]["T_w_c"], dtype=np.float64)
        GT_T = np.asarray(r["GT"]["T_w_c"], dtype=np.float64)

        r["metrics"].update({
            "is_pred_pointcloud_empty": int(pred_pts.shape[0] == 0),
            "num_pred_pointcloud": pred_pts.shape[0],
            "num_input_pointcloud": input_pc.shape[0],
            "num_GT_pointcloud": GT_pc.shape[0],
        })

        if pred_pts.shape[0] > 0:
            pred_chunks.append(transform_points(pred_pts.astype(np.float64), pred_T))
            pred_color_chunks.append(r["GT"]["undistorted_raw_image"][pred_mask].astype("f4") / 255.0)
        if input_pc.shape[0] > 0:
            input_chunks.append(transform_points(input_pc.astype(np.float64), GT_T))
        if GT_pc.shape[0] > 0:
            GT_chunks.append(transform_points(GT_pc.astype(np.float64), GT_T))

    all_pred = _stack(pred_chunks)
    all_input = _stack(input_chunks)
    all_GT = _stack(GT_chunks)
    all_pred_colors = _stack(pred_color_chunks)

    # ---- Scene-level point-cloud metrics (computed once) ----
    scene_pc_metrics = _scene_pc_metrics(all_GT, all_input, all_pred)

    # ---- Save metrics.json + .pcd + depth PNGs + pose.txt ----
    saveMetricsLogAndResults(config, result_list, scene_pc_metrics, all_pred, all_pred_colors)
    print("end")


@hydra.main(
    version_base=None, config_path="../../../configs", config_name="dense_n_view_benchmark"
)
def execute_dense_n_view_benchmark(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    dense_n_view_benchmark(cfg)


if __name__ == "__main__":
    execute_dense_n_view_benchmark()
