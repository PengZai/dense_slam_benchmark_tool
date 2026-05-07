import torch
from mapanything.models import MapAnything
from mapanything.utils.geometry import quaternion_to_rotation_matrix
import time


class MapAnythingWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        isInputIntrinsics,
        isInputCameraPoses,
        isInputDepthZ,
        **kwargs,
    ):
        super().__init__()

        self.name = name

        self.isInputIntrinsics = isInputIntrinsics
        self.isInputCameraPoses = isInputCameraPoses
        self.isInputDepthZ = isInputDepthZ


        self.model = MapAnything(name, **kwargs)

    
    def forward(self, frames):

        # convert multi camera in single frame to views
        num_frame = len(frames)

        num_views_per_frame = len(frames[0])
        views = [view for frame in frames for view in frame]

        batch_size_per_view, _, height, width = views[0]["undistorted_image"].shape

        input_views = []
        for view in views:


            input_view = {
                'img':view['undistorted_image'],
                "data_norm_type": view['data_norm_type'],
                'is_metric_scale': view['is_metric_scale'], 
            }

            if self.isInputIntrinsics:
                input_view['intrinsics'] = view['intrinsics']
            if self.isInputCameraPoses:
                input_view['camera_poses'] = view['T_w_c']
            if self.isInputDepthZ:
                input_view['depth_z'] = view['input_depth'].squeeze(1)


            input_views.append(input_view)

        start = time.time()

        outputs = self.model.infer(
                input_views,
                memory_efficient_inference=True,
                minibatch_size=batch_size_per_view,
                ignore_calibration_inputs=False,  # Whether to use COLMAP calibration or not
                ignore_depth_inputs=False,  # COLMAP doesn't provide depth (can recover from sparse points but convoluted)
                ignore_pose_inputs=False,  # Whether to use COLMAP poses or not
                ignore_depth_scale_inputs=False,  # No depth data
                ignore_pose_scale_inputs=False,  # COLMAP poses are non-metric
                # Use amp for better performance
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
            )
        
        end = time.time()

        runtime = end - start
    
        res = []
        for frame_idx in range(num_frame):

            pred_idx = frame_idx*num_views_per_frame
            depth_z = outputs[pred_idx]['depth_z'].squeeze(-1).detach().cpu().numpy()
            mask = outputs[pred_idx]['mask'].squeeze(-1).detach().cpu().numpy()
            valid_mask = depth_z > 0.0
            depth_confidence = None
            for confidence_key in ("conf", "depth_conf", "depth_confidence", "confidence"):
                if confidence_key in outputs[pred_idx]:
                    depth_confidence = (
                        outputs[pred_idx][confidence_key]
                        .squeeze(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    break
            if depth_confidence is None:
                depth_confidence = (mask & valid_mask).astype("float32")
        
            pred_T_w_c = torch.eye(4, device=outputs[pred_idx]["cam_quats"].device).unsqueeze(0)
            pred_T_w_c_rot = quaternion_to_rotation_matrix(outputs[pred_idx]["cam_quats"].clone())
            pred_T_w_c[..., :3, :3] = pred_T_w_c_rot
            pred_T_w_c[..., :3, 3] = outputs[pred_idx]["cam_trans"].clone()
            pred_T_w_c = pred_T_w_c.cpu().numpy()

            pred_intrinsics = None
            for k_key in ("intrinsics", "K", "camera_intrinsics"):
                if k_key in outputs[pred_idx]:
                    pred_intrinsics = outputs[pred_idx][k_key].detach().cpu().numpy()
                    break
            if pred_intrinsics is None:
                pred_intrinsics = views[pred_idx]['intrinsics'].cpu().numpy()
            res.append(
                {
                    'pred_depth':depth_z,
                    'pred_depth_mask': mask & valid_mask,  # this 1 threshold according to scene.show() visualization setting
                    'pred_depth_confidence': depth_confidence,
                    'pred_T_w_c': pred_T_w_c,
                    'pred_intrinsics': pred_intrinsics,
                    'runtime': runtime/float(num_frame)
                }
            )

        return res
