# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for Depth Anything 3
"""

import time

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3


class DA3Wrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        hf_model_name,
        isInputIntrinsics=False,
        isInputCameraPoses=False,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.isInputIntrinsics = isInputIntrinsics
        self.isInputCameraPoses = isInputCameraPoses
        self.model = DepthAnything3.from_pretrained(hf_model_name)

    @staticmethod
    def _as_world_to_camera(T_w_c):
        return torch.linalg.inv(T_w_c)

    @staticmethod
    def _as_camera_to_world(extrinsics):
        if extrinsics is None:
            return None

        if extrinsics.shape[-2:] == (3, 4):
            bottom = np.broadcast_to(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=extrinsics.dtype),
                (*extrinsics.shape[:-2], 1, 4),
            )
            extrinsics = np.concatenate([extrinsics, bottom], axis=-2)

        return np.linalg.inv(extrinsics)

    def _convert_output_to_prediction(self, output):
        if hasattr(self.model, "_convert_to_prediction"):
            return self.model._convert_to_prediction(output)
        return self.model.output_processor(output)

    @staticmethod
    def _prediction_confidence(prediction, depth):
        for name in ("conf", "depth_conf", "depth_confidence", "confidence"):
            value = getattr(prediction, name, None)
            if value is not None:
                value = np.asarray(value)
                if value.ndim == depth.ndim + 1 and value.shape[-1] == 1:
                    value = value.squeeze(-1)
                return value
        return (np.isfinite(depth) & (depth > 0)).astype(np.float32)

    def forward(self, frames):
        num_frame = len(frames)
        num_views_per_frame = len(frames[0])
        views = [view for frame in frames for view in frame]

        images = torch.stack([view["undistorted_image"] for view in views], dim=1)
        batch_size = images.shape[0]

        intrinsics = None
        if self.isInputIntrinsics:
            intrinsics = torch.stack([view["intrinsics"] for view in views], dim=1).float()

        original_extrinsics = None
        if self.isInputCameraPoses:
            T_w_c = torch.stack([view["T_w_c"] for view in views], dim=1).float()
            original_extrinsics = self._as_world_to_camera(T_w_c)

        if images.device.type == "cuda":
            torch.cuda.synchronize(images.device)
        start = time.time()

        batch_depths = []
        batch_depth_confidences = []
        batch_poses = []
        batch_intrinsics_pred = []
        for batch_idx in range(batch_size):
            batch_images = images[batch_idx : batch_idx + 1]
            batch_intrinsics = (
                intrinsics[batch_idx : batch_idx + 1] if intrinsics is not None else None
            )
            batch_original_extrinsics = (
                original_extrinsics[batch_idx : batch_idx + 1]
                if original_extrinsics is not None
                else None
            )
            batch_extrinsics = batch_original_extrinsics
            if batch_extrinsics is not None and hasattr(self.model, "_normalize_extrinsics"):
                batch_extrinsics = self.model._normalize_extrinsics(batch_extrinsics.clone())

            output = self.model.forward(
                batch_images,
                extrinsics=batch_extrinsics,
                intrinsics=batch_intrinsics,
                export_feat_layers=[],
            )
            prediction = self._convert_output_to_prediction(output)

            if batch_original_extrinsics is not None and hasattr(
                self.model, "_align_to_input_extrinsics_intrinsics"
            ):
                prediction = self.model._align_to_input_extrinsics_intrinsics(
                    batch_original_extrinsics.squeeze(0).detach().cpu(),
                    batch_intrinsics.squeeze(0).detach().cpu()
                    if batch_intrinsics is not None
                    else None,
                    prediction,
                )

            batch_depths.append(prediction.depth)
            batch_depth_confidences.append(
                self._prediction_confidence(prediction, prediction.depth)
            )
            if getattr(prediction, "extrinsics", None) is not None:
                batch_poses.append(self._as_camera_to_world(prediction.extrinsics))
            else:
                batch_poses.append(None)
            pred_intr = getattr(prediction, "intrinsics", None)
            if pred_intr is not None:
                pred_intr = np.asarray(pred_intr)
            batch_intrinsics_pred.append(pred_intr)

        if images.device.type == "cuda":
            torch.cuda.synchronize(images.device)
        runtime = time.time() - start

        depth = np.stack(batch_depths, axis=0)
        depth_confidence = np.stack(batch_depth_confidences, axis=0)
        pred_T_w_c = None
        if all(batch_pose is not None for batch_pose in batch_poses):
            pred_T_w_c = np.stack(batch_poses, axis=0)
        pred_intrinsics = None
        if all(batch_intr is not None for batch_intr in batch_intrinsics_pred):
            pred_intrinsics = np.stack(batch_intrinsics_pred, axis=0)

        results = []
        for frame_idx in range(num_frame):
            pred_idx = frame_idx * num_views_per_frame
            pred_depth = depth[:, pred_idx]
            pred_depth_mask = np.isfinite(pred_depth) & (pred_depth > 0)

            if pred_T_w_c is None:
                frame_T_w_c = frames[frame_idx][0]["T_w_c"].detach().cpu().numpy()
            else:
                frame_T_w_c = pred_T_w_c[:, pred_idx]

            if pred_intrinsics is None:
                frame_intrinsics = frames[frame_idx][0]["intrinsics"].detach().cpu().numpy()
            else:
                frame_intrinsics = pred_intrinsics[:, pred_idx]

            results.append(
                {
                    "pred_depth": pred_depth,
                    "pred_depth_mask": pred_depth_mask,
                    "pred_depth_confidence": depth_confidence[:, pred_idx],
                    "pred_T_w_c": frame_T_w_c,
                    "pred_intrinsics": frame_intrinsics,
                    "runtime": runtime / float(num_frame),
                }
            )

        return results
