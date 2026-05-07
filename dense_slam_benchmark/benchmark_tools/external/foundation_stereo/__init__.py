import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder


class FoundationStereoWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_dir,
        valid_iters=12,
        hiera=False,
        small_ratio=0.5,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.valid_iters = valid_iters
        self.hiera = bool(hiera)
        self.small_ratio = small_ratio

        cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_dir), "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = valid_iters
        cfg["hiera"] = int(self.hiera)
        args = OmegaConf.create(cfg)

        self.model = FoundationStereo(args)
        ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])

    @staticmethod
    def _stereo_baseline(T_w_l, T_w_r):
        # Baseline = x-axis component of right-camera origin in the left-camera frame.
        T_l_r = np.linalg.inv(T_w_l) @ T_w_r
        return float(abs(T_l_r[0, 3]))

    def forward(self, frames):
        device = next(self.model.parameters()).device
        results = []

        for frame in frames:
            assert len(frame) >= 2, "FoundationStereo requires a stereo pair (left, right) per frame"
            left_view, right_view = frame[0], frame[1]

            left_raw = left_view["undistorted_raw_image"]   # (B, H, W, 3) BGR uint8 (torch or numpy)
            right_raw = right_view["undistorted_raw_image"]
            if not isinstance(left_raw, torch.Tensor):
                left_raw = torch.as_tensor(left_raw)
                right_raw = torch.as_tensor(right_raw)
            B, H, W, _ = left_raw.shape

            # cv2 layout BGR -> RGB; (B, H, W, 3) -> (B, 3, H, W) float32 on device
            left_t = left_raw.flip(-1).permute(0, 3, 1, 2).to(
                dtype=torch.float32, device=device
            )
            right_t = right_raw.flip(-1).permute(0, 3, 1, 2).to(
                dtype=torch.float32, device=device
            )

            padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
            left_t, right_t = padder.pad(left_t, right_t)

            start = time.time()
            with torch.cuda.amp.autocast(True):
                if not self.hiera:
                    disp = self.model.forward(
                        left_t, right_t, iters=self.valid_iters, test_mode=True
                    )
                else:
                    disp = self.model.run_hierachical(
                        left_t, right_t,
                        iters=self.valid_iters, test_mode=True,
                        small_ratio=self.small_ratio,
                    )
            disp = padder.unpad(disp.float()).reshape(B, H, W)
            disp_np = disp.detach().cpu().numpy()
            # print(f"{left_view['ts'].item()}, disp sum = {np.sum(np.abs(disp_np - disp_np.mean()))}")
            runtime = time.time() - start

            K_left = left_view["intrinsics"].cpu().numpy()   # (B, 3, 3)
            T_w_l = left_view["T_w_c"].cpu().numpy()         # (B, 4, 4)
            T_w_r = right_view["T_w_c"].cpu().numpy()

            # depth = fx * baseline / disp, per batch element.
            depths = np.zeros((B, H, W), dtype=np.float32)
            masks = np.zeros((B, H, W), dtype=bool)
            for b in range(B):
                fx = K_left[b, 0, 0]
                baseline = self._stereo_baseline(T_w_l[b], T_w_r[b])
                d = disp_np[b]
                valid = d > 1e-3
                depth = np.zeros_like(d, dtype=np.float32)
                depth[valid] = fx * baseline / d[valid]
                depths[b] = depth
                masks[b] = valid


            # print(f"{left_view['ts'].item()}, depth sum = {np.sum(np.abs(depths[masks] - depths[masks].mean()))}, fx * baseline :{fx * baseline} ")

            results.append({
                "pred_depth": depths,
                "pred_depth_mask": masks,
                "pred_depth_confidence": masks.astype("float32"),
                "pred_T_w_c": T_w_l,
                "pred_intrinsics": K_left,
                "runtime": runtime,
            })

        return results
