import importlib
import inspect
import time

import torch


def _resolve_depth_anything_v2_class(metric):
    """Locate the DepthAnythingV2 class to use.

    For relative depth: always `depth_anything_v2.dpt.DepthAnythingV2`.

    For metric depth: try a metric-specific submodule first (some forks ship
    upstream's `metric_depth/depth_anything_v2/dpt.py` under one of these names),
    then fall back to the relative-class location IFF its constructor accepts
    `max_depth`. Otherwise raise a clear error pointing at the upstream metric
    repo so the user knows what's missing.
    """
    if not metric:
        from depth_anything_v2.dpt import DepthAnythingV2
        return DepthAnythingV2

    candidate_paths = (
        "depth_anything_v2.metric.dpt",
        "depth_anything_v2.metric_depth.dpt",
        "depth_anything_v2_metric.dpt",
        "depth_anything_v2.dpt",  # last resort: unified-class forks
    )
    for path in candidate_paths:
        try:
            mod = importlib.import_module(path)
        except ImportError:
            continue
        cls = getattr(mod, "DepthAnythingV2", None)
        if cls is None:
            continue
        if "max_depth" in inspect.signature(cls.__init__).parameters:
            return cls

    raise RuntimeError(
        "metric=True requires a DepthAnythingV2 class that accepts a max_depth "
        "kwarg. None of the candidate import paths "
        f"{list(candidate_paths)} expose one. Install the metric variant from "
        "https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth "
        "(e.g. update the PengZai/Depth-Anything-V2 fork to package the "
        "`metric_depth/depth_anything_v2/dpt.py` class as a sibling module)."
    )


class DepthAnythingV2Wrapper(torch.nn.Module):
    # Default max_depth (meters) per fine-tuned metric checkpoint, taken from
    # https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth
    METRIC_DATASET_DEFAULT_MAX_DEPTH = {"hypersim": 20.0, "vkitti": 80.0}

    def __init__(
        self,
        name,
        ckpt_dir,
        encoder,
        metric=False,
        metric_dataset="hypersim",
        max_depth=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }

        self.metric = bool(metric)
        Cls = _resolve_depth_anything_v2_class(self.metric)

        if self.metric:
            if metric_dataset not in self.METRIC_DATASET_DEFAULT_MAX_DEPTH:
                raise ValueError(
                    f"Unknown metric_dataset '{metric_dataset}'. Expected one of "
                    f"{sorted(self.METRIC_DATASET_DEFAULT_MAX_DEPTH)}."
                )
            self.metric_dataset = metric_dataset
            self.max_depth = float(
                max_depth if max_depth is not None
                else self.METRIC_DATASET_DEFAULT_MAX_DEPTH[metric_dataset]
            )
            self.model = Cls(
                **{**model_configs[encoder], "max_depth": self.max_depth}
            )
            ckpt_path = f"{ckpt_dir}/depth_anything_v2_metric_{metric_dataset}_{encoder}.pth"
        else:
            self.metric_dataset = None
            self.max_depth = None
            self.model = Cls(**model_configs[encoder])
            ckpt_path = f"{ckpt_dir}/depth_anything_v2_{encoder}.pth"

        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    def forward(self, frames):
        views = [view for frame in frames for view in frame]

        res = []
        for view in views:
            input_view = {"x": view["undistorted_image"]}

            start = time.time()
            output = self.model.forward(**input_view)

            if self.metric:
                # Metric variant emits depth in meters in [0, max_depth].
                depth = output
                mask = depth > 1e-3
            else:
                # Relative variant emits inverse depth; convert to depth.
                inv_depth = output
                mask = inv_depth > 1e-3
                depth = torch.zeros_like(inv_depth)
                depth[mask] = 1.0 / inv_depth[mask]

            depth = depth.detach().cpu().squeeze(1).numpy()
            end = time.time()
            runtime = end - start

            res.append({
                "pred_depth": depth,
                "pred_depth_mask": depth > 0,
                "pred_depth_confidence": (depth > 0).astype("float32"),
                "pred_T_w_c": view["T_w_c"].cpu().numpy(),
                "pred_intrinsics": view["intrinsics"].cpu().numpy(),
                "runtime": runtime,
            })

        return res
