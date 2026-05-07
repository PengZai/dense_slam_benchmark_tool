"""
Inference wrapper for VGGT (Visual Geometry Grounded Transformer)
"""

import time

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        hf_model_name,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.model = VGGT.from_pretrained(hf_model_name)

    @staticmethod
    def _world_to_camera_3x4_to_camera_to_world_4x4(extrinsics_3x4):
        S = extrinsics_3x4.shape[0]
        bottom = np.broadcast_to(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=extrinsics_3x4.dtype),
            (S, 1, 4),
        )
        homo = np.concatenate([extrinsics_3x4, bottom], axis=-2)
        return np.linalg.inv(homo)

    def forward(self, frames):
        num_frame = len(frames)
        num_views_per_frame = len(frames[0])
        views = [view for frame in frames for view in frame]

        images = torch.stack([view["undistorted_image"] for view in views], dim=1)
        batch_size, S, _, H, W = images.shape

        if images.device.type == "cuda":
            torch.cuda.synchronize(images.device)
        start = time.time()

        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        batch_depths = []
        batch_depth_confidences = []
        batch_poses = []
        batch_intrinsics = []
        for batch_idx in range(batch_size):
            batch_images = images[batch_idx : batch_idx + 1]

            with torch.no_grad():
                with torch.autocast(
                    device_type=batch_images.device.type, dtype=autocast_dtype
                ):
                    output = self.model(batch_images)

            pose_enc = output["pose_enc"].float()
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(H, W), build_intrinsics=True
            )
            extrinsics = extrinsics.detach().cpu().numpy().squeeze(0)
            intrinsics = intrinsics.detach().cpu().numpy().squeeze(0)

            depth = output["depth"].float().detach().cpu().numpy()
            depth = depth.squeeze(0).squeeze(-1)
            if "depth_conf" in output:
                depth_confidence = output["depth_conf"].float().detach().cpu().numpy()
                depth_confidence = depth_confidence.squeeze(0)
                if depth_confidence.ndim == 4 and depth_confidence.shape[-1] == 1:
                    depth_confidence = depth_confidence.squeeze(-1)
            else:
                depth_confidence = (np.isfinite(depth) & (depth > 0)).astype(np.float32)

            T_w_c = self._world_to_camera_3x4_to_camera_to_world_4x4(extrinsics)

            batch_depths.append(depth)
            batch_depth_confidences.append(depth_confidence)
            batch_poses.append(T_w_c)
            batch_intrinsics.append(intrinsics)

        if images.device.type == "cuda":
            torch.cuda.synchronize(images.device)
        runtime = time.time() - start

        depth = np.stack(batch_depths, axis=0)
        depth_confidence = np.stack(batch_depth_confidences, axis=0)
        pred_T_w_c = np.stack(batch_poses, axis=0)
        pred_intrinsics = np.stack(batch_intrinsics, axis=0)

        results = []
        for frame_idx in range(num_frame):
            view_idx = frame_idx * num_views_per_frame
            pred_depth = depth[:, view_idx]
            pred_depth_mask = np.isfinite(pred_depth) & (pred_depth > 0)

            results.append(
                {
                    "pred_depth": pred_depth,
                    "pred_depth_mask": pred_depth_mask,
                    "pred_depth_confidence": depth_confidence[:, view_idx],
                    "pred_T_w_c": pred_T_w_c[:, view_idx],
                    "pred_intrinsics": pred_intrinsics[:, view_idx],
                    "runtime": runtime / float(num_frame),
                }
            )

        return results
