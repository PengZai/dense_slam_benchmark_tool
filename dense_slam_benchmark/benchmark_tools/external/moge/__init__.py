import time

import numpy as np
import torch

from moge.model import import_model_class_by_version


class MoGeWrapper(torch.nn.Module):
    # Source: thirdparty/MoGe/moge/scripts/infer.py
    DEFAULT_PRETRAINED = {
        "v1": "Ruicheng/moge-vitl",
        "v2": "Ruicheng/moge-2-vitl-normal",
    }

    def __init__(
        self,
        name,
        version="v2",
        pretrained_model_name_or_path=None,
        resolution_level=9,
        num_tokens=None,
        use_fp16=True,
        use_input_fov_x=False,
        **kwargs,
    ):
        super().__init__()
        if version not in self.DEFAULT_PRETRAINED:
            raise ValueError(
                f"Unknown MoGe version '{version}'. "
                f"Expected one of {sorted(self.DEFAULT_PRETRAINED)}."
            )
        self.name = name
        self.version = version
        self.resolution_level = int(resolution_level)
        self.num_tokens = num_tokens  # None lets the model pick from resolution_level
        self.use_fp16 = bool(use_fp16)
        self.use_input_fov_x = bool(use_input_fov_x)

        ckpt = pretrained_model_name_or_path or self.DEFAULT_PRETRAINED[version]
        Cls = import_model_class_by_version(version)
        self.model = Cls.from_pretrained(ckpt)

    @staticmethod
    def _intrinsics_norm_to_pixel(K_norm, height, width):
        """MoGe returns intrinsics in normalized image coords (cx=cy=0.5,
        fx/fy in [0,1]). Convert to pixel-space so make_pts3d in the benchmark
        postprocess works as expected."""
        K_pix = K_norm.copy()
        K_pix[..., 0, 0] *= width   # fx
        K_pix[..., 0, 2] *= width   # cx
        K_pix[..., 1, 1] *= height  # fy
        K_pix[..., 1, 2] *= height  # cy
        return K_pix

    def forward(self, frames):
        device = next(self.model.parameters()).device

        # Monocular: flatten (frame, view) into a flat list — same pattern as
        # the depth_anything_v2 wrapper. Assumes one camera per frame.
        views = [view for frame in frames for view in frame]

        results = []
        for view in views:
            raw = view["undistorted_raw_image"]  # (B, H, W, 3) BGR uint8 (torch or numpy)
            if not isinstance(raw, torch.Tensor):
                raw = torch.as_tensor(raw)
            B, H, W, _ = raw.shape

            # cv2 layout BGR -> RGB; (B, H, W, 3) -> (B, 3, H, W); uint8 [0,255] -> float32 [0,1]
            image_t = (
                raw.flip(-1).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
                / 255.0
            )

            fov_x = None
            if self.use_input_fov_x:
                K_in = view["intrinsics"]
                if isinstance(K_in, torch.Tensor):
                    K_in = K_in.cpu().numpy()
                K_in = np.asarray(K_in)  # (B, 3, 3)
                fx_pix = float(K_in[0, 0, 0])
                # fov_x = 2 * atan(W / (2 * fx)), in degrees as MoGe expects
                fov_x = float(np.rad2deg(2.0 * np.arctan(W / (2.0 * fx_pix))))

            start = time.time()
            with torch.no_grad():
                out = self.model.infer(
                    image_t,
                    fov_x=fov_x,
                    num_tokens=self.num_tokens,
                    resolution_level=self.resolution_level,
                    use_fp16=self.use_fp16,
                    apply_mask=False,  # keep raw depth; mask comes back separately
                )
            runtime = time.time() - start

            depth = out["depth"].detach().cpu().numpy().astype(np.float32)  # (B, H, W)
            if "mask" in out:
                mask = out["mask"].detach().cpu().numpy().astype(bool)      # (B, H, W)
            else:
                mask = depth > 0
            mask &= np.isfinite(depth) & (depth > 0)
            depth[~mask] = 0.0

            K_norm = out["intrinsics"].detach().cpu().numpy()                # (B, 3, 3) normalized
            K_pix = self._intrinsics_norm_to_pixel(K_norm, H, W).astype(np.float32)

            T_w_c = view["T_w_c"]
            if isinstance(T_w_c, torch.Tensor):
                T_w_c = T_w_c.cpu().numpy()
            T_w_c = np.asarray(T_w_c)

            results.append({
                "pred_depth": depth,
                "pred_depth_mask": mask,
                "pred_depth_confidence": mask.astype(np.float32),
                "pred_T_w_c": T_w_c,
                "pred_intrinsics": K_pix,
                "runtime": runtime,
            })

        return results
