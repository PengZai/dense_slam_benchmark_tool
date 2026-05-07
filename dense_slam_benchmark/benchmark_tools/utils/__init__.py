import json
import os
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

from dense_slam_benchmark.dataset_tools.utils import (
    T_to_pose,
    depth2color,
    depth_range_by_ratio,
)


SUM_KEYS = {"runtime", "postprocess_time"}


def _make_unique_dir(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path
    idx = 1
    while True:
        new_path = f"{base_path}_{idx}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        idx += 1


def _aggregate_per_frame_metrics(result_list):
    basic = defaultdict(list)
    per_frame = defaultdict(list)
    for r in result_list:
        for k, v in r["basic"].items():
            basic[k].append(v)
        for k, v in r["metrics"].items():
            per_frame[k].append(float(v))

    aggregated = {
        k: float(np.sum(v) if k in SUM_KEYS else np.mean(v))
        for k, v in per_frame.items()
    }
    return dict(basic), aggregated


def _save_depth_images(result_list, output_path):
    os.makedirs(output_path, exist_ok=True)
    for r in result_list:
        pred_depth = r["pred"]["depth"].squeeze()
        depth_range = depth_range_by_ratio(pred_depth, keep=0.98)
        vis = depth2color(pred_depth, min_depth=depth_range[0], max_depth=depth_range[1])
        cv2.imwrite(
            os.path.join(output_path, f"pred_depth_{r['basic']['sample_idx']}.png"),
            vis,
        )


def _save_pred_pointcloud(points, colors, output_path, filename="pred_scene.pcd"):
    if points.shape[0] == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    if colors.shape[0] == points.shape[0]:
        # BGR (cv2 layout) -> RGB
        pcd.colors = o3d.utility.Vector3dVector(colors[:, [2, 1, 0]].astype(np.float64))
    o3d.io.write_point_cloud(os.path.join(output_path, filename), pcd, write_ascii=False)


def _format_ts(ts):
    if isinstance(ts, str):
        return ts
    return f"{float(ts):.9f}"


def _save_poses(result_list, output_path):
    pose_path = os.path.join(output_path, "pred_pose.txt")
    with open(pose_path, "w") as f:
        f.write("#timestamp/index x y z q_x q_y q_z q_w\n")
        for r in result_list:
            T = np.asarray(r["pred"]["T_w_c"], dtype=np.float64)
            if T.ndim == 3:
                T = T[0]
            t, q = T_to_pose(T)
            ts = r["basic"].get("ts", r["basic"]["sample_idx"])
            f.write(
                f"{_format_ts(ts)} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} "
                f"{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f}\n"
            )



def saveMetricsLogAndResults(config, result_list, scene_pc_metrics, pred_pc_world, pred_pc_colors):
    output_path = _make_unique_dir(os.path.join(
        config.machine.root_experiments_dir,
        f"dense_{config.num_view_for_sub_scene}_view_stride_{config.stride_for_sub_scene}",
        config.model.model_str,
    ))
    depth_image_dir = os.path.join(output_path, "depth_per_sample")

    basic, aggregated = _aggregate_per_frame_metrics(result_list)
    summary = {
        "basic": basic,
        "metrics": {**aggregated, **scene_pc_metrics},
    }

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=4)
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4, default=str)

    _save_depth_images(result_list, depth_image_dir)
    _save_pred_pointcloud(pred_pc_world, pred_pc_colors, output_path)
    _save_poses(result_list, output_path)


def updateConfig(config, camera_configs):
    config["cameras"] = {}
    for camera_id in config["used_camera_idx_per_view"]:
        config["cameras"]["camera" + str(camera_id)] = camera_configs["camera" + str(camera_id)]
        camera_name = config["cameras"]["camera" + str(camera_id)]["name"]
        config["cameras"]["camera" + str(camera_id)]["datapath"] = {
            "undistorted_images": os.path.join(config["root_data_dir"], config["scene_name"], camera_name, "undistorted_images"),
            "input_pose": os.path.join(config["root_data_dir"], config["scene_name"], camera_name, config["input_pose_name"]),
            "input_depth": os.path.join(config["root_data_dir"], config["scene_name"], camera_name, config["input_geometry_dir"], "depth"),
            "input_pointcloud": os.path.join(config["root_data_dir"], config["scene_name"], camera_name, config["input_geometry_dir"], "pointcloud"),
            "GT_depth": os.path.join(config["root_data_dir"], config["scene_name"], camera_name, config["GT_geometry_dir_name"], "depth"),
            "GT_pointcloud": os.path.join(config["root_data_dir"], config["scene_name"], camera_name, config["GT_geometry_dir_name"], "pointcloud"),
        }
    return config
