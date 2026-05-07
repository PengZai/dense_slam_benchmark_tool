from contextlib import contextmanager
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dense_slam_benchmark.dataset_tools.utils import T_to_pose


class HLocWrapper(torch.nn.Module):
    SPARSE_MODULES = {"superpoint", "sift", "loma"}
    DENSE_MODULES = {"loftr", "efficient_loftr", "roma", "romav2"}

    FEATURE_DEFAULTS = {
        "superpoint": "superpoint_max",
        "sift": "sift",
        "loma": "loma_aachen",
    }
    MATCHER_DEFAULTS = {
        "superpoint": "superglue",
        "sift": "NN-mutual",
        "loma": "loma",
    }

    # User-facing camera_mode values, mapped to pycolmap.CameraMode names.
    # "single"     : every image shares one Camera record (one set of intrinsics).
    # "per_folder" : one Camera per cam{NN}/ subfolder; images in the same camera
    #                share intrinsics but different cameras have their own.
    #                Required by use_rig=True (apply_rig_config groups images
    #                by image_prefix and demands a single camera_id per prefix).
    # "per_image"  : every image gets its own Camera record (no sharing).
    # "auto"       : let COLMAP decide based on EXIF / heuristics.
    CAMERA_MODE_NAMES = {"single", "per_folder", "per_image", "auto"}

    # Pipeline switch:
    #   "reconstruction"          : full incremental SfM via hloc.reconstruction.main
    #   "triangulate_with_poses"  : triangulate from input poses (+ optional BA)
    #                               via hloc.triangulate_with_poses.main
    PIPELINE_NAMES = {"reconstruction", "triangulate_with_poses"}

    # Which images get a position prior when use_input_poses_as_prior is on:
    #   "all"         : every image's input T_w_c becomes a soft BA constraint.
    #   "first_frame" : only views at frame_idx == 0 are anchored (equivalent
    #                   to N=1 below; kept as a named alias).
    #   positive int N: views at frame_idx < N are anchored, the rest refine
    #                   freely. Useful as a gauge-fix or to seed a head section.
    POSE_PRIOR_SCOPES_STR = {"all", "first_frame"}

    def __init__(
        self,
        name,
        cache_dir,
        keep_cache=True,
        visualize_pred_pointcloud=False,
        show_pred_pointcloud=False,
        pred_pointcloud_dir=None,
        pred_pointcloud_max_points=1000000,
        module="superpoint",
        feature_conf=None,
        matcher_conf=None,
        dense_matcher_conf=None,
        pair_mode="sequential",
        sequential_overlap=10,
        use_input_intrinsics=True,
        pipeline="reconstruction",
        skip_bundle_adjustment=True,
        no_refine_intrinsics=True,
        skip_geometric_verification=False,
        min_model_size=2,
        n_threads=16,
        max_kps=8192,
        verbose=False,
        triangulation_options=None,
        bundle_adjustment_options=None,
        use_rig=False,
        camera_mode="auto",
        use_input_poses_as_prior=False,
        pose_prior_scope="all",
        prior_position_loss_scale=7.815,
        prior_use_robust_loss=False,
        prior_position_sigma=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.use_rig = use_rig
        if camera_mode not in self.CAMERA_MODE_NAMES:
            raise ValueError(
                f"Unknown camera_mode '{camera_mode}'. "
                f"Expected one of {sorted(self.CAMERA_MODE_NAMES)}."
            )
        if use_rig and camera_mode not in ("per_folder", "auto"):
            print(
                f"[{name}] use_rig=True forces camera_mode='per_folder'; "
                f"requested '{camera_mode}' will be overridden."
            )
            camera_mode = "per_folder"
        self.camera_mode = camera_mode
        self.cache_dir = Path(cache_dir)
        self.keep_cache = keep_cache
        self.visualize_pred_pointcloud = visualize_pred_pointcloud
        self.show_pred_pointcloud = show_pred_pointcloud
        self.pred_pointcloud_dir = (
            Path(pred_pointcloud_dir) if pred_pointcloud_dir is not None else None
        )
        self.pred_pointcloud_max_points = pred_pointcloud_max_points
        module = "efficient_loftr" if module == "efficientloftr" else module
        self.module = module
        self.feature_conf = feature_conf or self.FEATURE_DEFAULTS.get(module)
        self.matcher_conf = matcher_conf or self.MATCHER_DEFAULTS.get(module)
        self.dense_matcher_conf = dense_matcher_conf or module
        self.pair_mode = pair_mode
        self.sequential_overlap = sequential_overlap
        self.use_input_intrinsics = use_input_intrinsics
        if pipeline not in self.PIPELINE_NAMES:
            raise ValueError(
                f"Unknown pipeline '{pipeline}'. "
                f"Expected one of {sorted(self.PIPELINE_NAMES)}."
            )
        if pipeline == "triangulate_with_poses" and not use_input_intrinsics:
            print(
                f"[{name}] pipeline='triangulate_with_poses' requires the input "
                "intrinsics + poses to be consistent; forcing "
                "use_input_intrinsics=True."
            )
            self.use_input_intrinsics = True
        self.pipeline = pipeline
        pose_prior_scope = self._validate_pose_prior_scope(pose_prior_scope)
        self.use_input_poses_as_prior = bool(use_input_poses_as_prior)
        self.pose_prior_scope = pose_prior_scope
        self.prior_position_loss_scale = float(prior_position_loss_scale)
        self.prior_use_robust_loss = bool(prior_use_robust_loss)
        self.prior_position_covariance = self._parse_position_sigma(prior_position_sigma)
        self.skip_bundle_adjustment = skip_bundle_adjustment
        self.no_refine_intrinsics = no_refine_intrinsics
        self.skip_geometric_verification = skip_geometric_verification
        self.min_model_size = min_model_size
        self.n_threads = n_threads
        self.max_kps = max_kps
        self.verbose = verbose
        self.triangulation_options = triangulation_options or {}
        self.bundle_adjustment_options = bundle_adjustment_options or {}

        if module not in self.SPARSE_MODULES | self.DENSE_MODULES:
            raise ValueError(
                f"Unsupported HLoc module '{module}'. "
                f"Expected one of {sorted(self.SPARSE_MODULES | self.DENSE_MODULES)}."
            )
        if pair_mode not in {"exhaustive", "sequential"}:
            raise ValueError("pair_mode must be either 'exhaustive' or 'sequential'.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _sample_work_dir(self):
        if self.keep_cache:
            yield Path(tempfile.mkdtemp(prefix=f"{self.name}_", dir=str(self.cache_dir)))
            return

        with tempfile.TemporaryDirectory(
            prefix=f"{self.name}_", dir=str(self.cache_dir)
        ) as tmpdir:
            yield Path(tmpdir)

    @staticmethod
    def _tensor_to_numpy(value, batch_idx=None):
        if torch.is_tensor(value):
            if batch_idx is not None and value.ndim > 0:
                value = value[batch_idx]
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value[batch_idx] if batch_idx is not None and value.ndim > 0 else value
        if isinstance(value, (list, tuple)):
            return value[batch_idx] if batch_idx is not None else value
        return value

    @staticmethod
    def _image_to_uint8(view, batch_idx):
        if "undistorted_raw_image" in view:
            image = HLocWrapper._tensor_to_numpy(view["undistorted_raw_image"], batch_idx)
            image = np.asarray(image)
        else:
            image = HLocWrapper._tensor_to_numpy(view["undistorted_image"], batch_idx)
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = np.transpose(image, (1, 2, 0))
            if image.dtype.kind == "f":
                image = np.clip(image, 0.0, 1.0) * 255.0

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image[..., :3])

    @staticmethod
    def _rigid3d_to_matrix(transform):
        matrix = getattr(transform, "matrix", None)
        if matrix is not None:
            if callable(matrix):
                matrix = matrix()
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape == (3, 4):
                bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
                matrix = np.concatenate([matrix, bottom], axis=0)
            return matrix

        rotation = getattr(transform, "rotation", None)
        translation = getattr(transform, "translation", None)
        if callable(rotation):
            rotation = rotation()
        if callable(translation):
            translation = translation()
        if hasattr(rotation, "matrix"):
            R = rotation.matrix()
            if callable(R):
                R = R()
        else:
            R = rotation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return T

    @classmethod
    def _cam_from_world_to_world_from_cam(cls, cam_from_world):
        if hasattr(cam_from_world, "inverse"):
            try:
                return cls._rigid3d_to_matrix(cam_from_world.inverse()).astype(np.float32)
            except Exception:
                pass
        T_c_w = cls._rigid3d_to_matrix(cam_from_world)
        return np.linalg.inv(T_c_w).astype(np.float32)

    def _view_name(self, frame_idx, view_idx):
        if self.use_rig:
            # subfolder per camera so each camera has its own image_prefix that
            # COLMAP can strip to recover the per-frame "remainder" used to group
            # images into Frames in apply_rig_config.
            return f"cam{view_idx:02d}/{frame_idx:06d}.png"
        return f"{frame_idx:06d}_{view_idx:02d}.png"

    def _camera_image_prefix(self, view_idx):
        return f"cam{view_idx:02d}/"

    def _build_rig_config(self, frames, batch_idx):
        """Build a single-rig RigConfig from the calibrated stereo poses on frame 0.

        Reference sensor is camera 0 (so the rig frame coincides with the left
        camera's frame). Each non-reference camera carries cam_from_rig =
        T_cam_from_world @ T_world_from_ref evaluated at frame 0.

        Returns None — and logs a one-shot warning — if rigs are enabled but the
        dataset only feeds one view per frame.
        """
        if not self.use_rig or not frames:
            return None
        if len(frames[0]) < 2:
            if not getattr(self, "_warned_rig_monocular", False):
                print(
                    f"[{self.name}] use_rig=True ignored: dataset has only "
                    f"{len(frames[0])} view(s) per frame; rig grouping needs >= 2."
                )
                self._warned_rig_monocular = True
            return None

        import pycolmap

        T_w_ref = self._tensor_to_numpy(
            frames[0][0]["T_w_c"], batch_idx
        ).astype(np.float64)

        rig_cameras = []
        for view_idx, view in enumerate(frames[0]):
            prefix = self._camera_image_prefix(view_idx)
            camera = self._rig_camera_from_view(view, batch_idx)

            if view_idx == 0:
                rig_cameras.append(
                    pycolmap.RigConfigCamera(
                        ref_sensor=True, image_prefix=prefix, camera=camera,
                    )
                )
                continue
            T_w_cam = self._tensor_to_numpy(view["T_w_c"], batch_idx).astype(np.float64)
            T_cam_rig = np.linalg.inv(T_w_cam) @ T_w_ref  # cam_from_ref
            rig_cameras.append(
                pycolmap.RigConfigCamera(
                    ref_sensor=False,
                    image_prefix=prefix,
                    camera=camera,
                    cam_from_rig=pycolmap.Rigid3d(matrix=T_cam_rig[:3, :4]),
                )
            )
        return [pycolmap.RigConfig(cameras=rig_cameras)]

    def _rig_camera_from_view(self, view, batch_idx):
        """Build a per-view pycolmap.Camera with this view's intrinsics + size.

        Returns None when intrinsics or image dimensions aren't recoverable;
        apply_rig_config will then fall back to whatever import_images created
        for that folder. Skips entirely when use_input_intrinsics is False so
        BA is free to estimate camera params.
        """
        if not self.use_input_intrinsics:
            return None
        K = self._tensor_to_numpy(view.get("intrinsics"), batch_idx)
        raw = view.get("undistorted_raw_image")
        if K is None or raw is None:
            return None
        K = np.asarray(K, dtype=np.float64)

        image = self._tensor_to_numpy(raw, batch_idx)
        if image is None or np.asarray(image).ndim < 2:
            return None
        h, w = np.asarray(image).shape[:2]

        import pycolmap

        return pycolmap.Camera(
            model="PINHOLE",
            width=int(w),
            height=int(h),
            params=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        )

    @classmethod
    def _validate_pose_prior_scope(cls, scope):
        """Accept "all", "first_frame", or a positive int (== first N frames).

        Returns the value normalized for storage:
          - the string "all" or "first_frame" passed through, or
          - a Python int (>= 1) for the integer form.
        """
        if isinstance(scope, bool):  # bool is a subclass of int; reject early
            raise ValueError(
                f"pose_prior_scope must be 'all', 'first_frame', or a positive "
                f"integer; got bool {scope!r}."
            )
        if isinstance(scope, str):
            if scope not in cls.POSE_PRIOR_SCOPES_STR:
                raise ValueError(
                    f"Unknown pose_prior_scope string '{scope}'. "
                    f"Expected one of {sorted(cls.POSE_PRIOR_SCOPES_STR)} "
                    "or a positive integer."
                )
            return scope
        if isinstance(scope, int):
            if scope <= 0:
                raise ValueError(
                    f"pose_prior_scope int must be a positive frame count "
                    f"(>=1); got {scope}."
                )
            return scope
        raise ValueError(
            f"pose_prior_scope must be 'all', 'first_frame', or a positive "
            f"integer; got {scope!r} of type {type(scope).__name__}."
        )

    def _pose_prior_prefix_length(self):
        """Resolve the configured scope to the number of leading frames to
        anchor. Returns None for "all" (meaning "no prefix cutoff")."""
        scope = self.pose_prior_scope
        if scope == "all":
            return None
        if scope == "first_frame":
            return 1
        return int(scope)

    @staticmethod
    def _parse_position_sigma(value):
        """Translate a user-facing sigma spec into a 3x3 covariance matrix.

        Accepts:
          - None                       -> None (covariance left unset on PosePrior)
          - positive scalar s          -> diag([s**2, s**2, s**2])
          - 3-sequence [sx, sy, sz]    -> diag([sx**2, sy**2, sz**2])

        Raises ValueError on anything else.
        """
        if value is None:
            return None
        if np.isscalar(value):
            s = float(value)
            if not (s > 0 and np.isfinite(s)):
                raise ValueError(
                    f"prior_position_sigma scalar must be a finite positive number; got {value!r}."
                )
            return np.diag([s * s] * 3).astype(np.float64)
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise ValueError(
                f"prior_position_sigma must be None, a positive scalar, or a "
                f"3-element sequence; got {value!r}."
            ) from exc
        if arr.shape != (3,) or not np.all(np.isfinite(arr)) or not np.all(arr > 0):
            raise ValueError(
                f"prior_position_sigma must be a length-3 sequence of finite "
                f"positive sigmas; got {value!r}."
            )
        return np.diag(arr * arr)

    def _build_pose_priors_dict(self, frames, batch_idx, metadata):
        """Build {image_name -> world position (3,) np.float64} for HLoc.

        Empty dict when use_input_poses_as_prior is False. The set of anchored
        images is determined by self.pose_prior_scope:
          - "all"            -> every (frame, view).
          - "first_frame"    -> views at frame_idx == 0 only.
          - positive int N   -> views at frame_idx < N (the first N frames).
        """
        if not self.use_input_poses_as_prior:
            return {}

        prefix = self._pose_prior_prefix_length()  # None means "no cutoff"
        priors = {}
        for name, meta in metadata.items():
            if prefix is not None and meta["frame_idx"] >= prefix:
                continue
            T = np.asarray(meta["T_w_c_input"], dtype=np.float64)
            priors[name] = T[:3, 3].copy()
        return priors

    def _prepare_sample(self, frames, batch_idx, image_dir, poses_path):
        image_list = []
        metadata = {}
        pose_lines = []

        for frame_idx, frame in enumerate(frames):
            for view_idx, view in enumerate(frame):
                name = self._view_name(frame_idx, view_idx)
                image_list.append(name)

                image = self._image_to_uint8(view, batch_idx)
                image_path = image_dir / name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise IOError(f"Failed to write temporary HLoc image {name}.")

                intrinsics = self._tensor_to_numpy(view["intrinsics"], batch_idx).astype(np.float64)
                T_w_c = self._tensor_to_numpy(view["T_w_c"], batch_idx).astype(np.float64)
                metadata[name] = {
                    "frame_idx": frame_idx,
                    "view_idx": view_idx,
                    "intrinsics": intrinsics,
                    "T_w_c_input": T_w_c,
                    "height": image.shape[0],
                    "width": image.shape[1],
                }

                T_c_w = np.linalg.inv(T_w_c)
                tvec, quat_xyzw = T_to_pose(T_c_w)
                pose_lines.append(
                    "{} {} {} {} {} {} {} {}".format(
                        name,
                        quat_xyzw[3],
                        quat_xyzw[0],
                        quat_xyzw[1],
                        quat_xyzw[2],
                        tvec[0],
                        tvec[1],
                        tvec[2],
                    )
                )

        poses_path.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
        return image_list, metadata

    def _build_pairs(self, outputs, image_list):
        from hloc import pairs_from_exhaustive

        pairs_path = outputs / f"pairs-{self.pair_mode}.txt"
        if self.pair_mode == "exhaustive":
            pairs_from_exhaustive.main(pairs_path, image_list=image_list)
            return pairs_path

        try:
            from hloc import pairs_from_sequential
        except ImportError as exc:
            raise ImportError(
                "pair_mode='sequential' requires hloc.pairs_from_sequential in the "
                "active benchmark-hloc environment."
            ) from exc
        pairs_from_sequential.main(
            pairs_path,
            image_list=image_list,
            overlap=self.sequential_overlap,
        )
        return pairs_path

    def _image_options(self, metadata, image_list):
        if not self.use_input_intrinsics:
            return None
        K = metadata[image_list[0]]["intrinsics"]
        return {
            "camera_model": "PINHOLE",
            "camera_params": f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}",
        }

    def _mapper_options(self):
        options = {"num_threads": self.n_threads}
        if self.no_refine_intrinsics:
            options.update(
                {
                    "ba_refine_focal_length": False,
                    "ba_refine_principal_point": False,
                    "ba_refine_extra_params": False,
                }
            )
        if self.use_input_poses_as_prior:
            options.update(
                {
                    "use_prior_position": True,
                    "use_robust_loss_on_prior_position": self.prior_use_robust_loss,
                    "prior_position_loss_scale": self.prior_position_loss_scale,
                }
            )
        return options

    def _run_hloc(self, image_dir, outputs, image_list, poses_path, metadata, rig_config=None, pose_priors=None, pose_prior_covariance=None):
        import pycolmap
        from hloc import reconstruction

        pairs_path = self._build_pairs(outputs, image_list)

        if self.module in self.SPARSE_MODULES:
            from hloc import extract_features, match_features

            if self.feature_conf not in extract_features.confs:
                raise KeyError(f"HLoc feature config '{self.feature_conf}' is not available.")
            if self.matcher_conf not in match_features.confs:
                raise KeyError(f"HLoc matcher config '{self.matcher_conf}' is not available.")
            feature_config = extract_features.confs[self.feature_conf]
            matcher_config = match_features.confs[self.matcher_conf]
            feature_path = extract_features.main(
                feature_config,
                image_dir,
                outputs,
                image_list=image_list,
                overwrite=True,
            )
            match_path = match_features.main(
                matcher_config,
                pairs_path,
                feature_config["output"],
                outputs,
                overwrite=True,
            )
        else:
            from hloc import match_dense

            if self.dense_matcher_conf not in match_dense.confs:
                raise KeyError(
                    f"HLoc dense matcher config '{self.dense_matcher_conf}' is not available."
                )
            feature_path, match_path = match_dense.main(
                match_dense.confs[self.dense_matcher_conf],
                pairs_path,
                image_dir,
                outputs,
                max_kps=self.max_kps,
                overwrite=True,
            )

        image_options = self._image_options(metadata, image_list)
        # apply_rig_config requires every image sharing a camera's image_prefix
        # (subfolder, in our naming scheme) to map to the same Camera record,
        # so use_rig=True is bound to "per_folder" at __init__ time. Otherwise
        # honor whatever the config asked for.
        effective_mode = "per_folder" if rig_config is not None else self.camera_mode
        camera_mode_lookup = {
            "auto": pycolmap.CameraMode.AUTO,
            "single": pycolmap.CameraMode.SINGLE,
            "per_folder": pycolmap.CameraMode.PER_FOLDER,
            "per_image": pycolmap.CameraMode.PER_IMAGE,
        }
        camera_mode = camera_mode_lookup[effective_mode]
        sfm_dir = outputs / f"sfm_{self.module}"

        if self.pipeline == "triangulate_with_poses":
            from hloc import triangulate_with_poses

            return triangulate_with_poses.main(
                sfm_dir,
                image_dir,
                pairs_path,
                feature_path,
                match_path,
                poses_path,
                camera_mode=camera_mode,
                verbose=self.verbose,
                skip_geometric_verification=self.skip_geometric_verification,
                image_list=image_list,
                image_options=image_options,
                triangulation_options=self.triangulation_options,
                bundle_adjustment_options=self.bundle_adjustment_options,
                refine=not self.skip_bundle_adjustment,
                rig_config=rig_config,
            )

        return reconstruction.main(
            sfm_dir,
            image_dir,
            pairs_path,
            feature_path,
            match_path,
            camera_mode=camera_mode,
            verbose=self.verbose,
            skip_geometric_verification=self.skip_geometric_verification,
            image_list=image_list,
            image_options=image_options,
            mapper_options=self._mapper_options(),
            rig_config=rig_config,
            pose_priors=pose_priors,
            pose_prior_covariance=pose_prior_covariance,
        )

    def _extract_image_pose(self, image, fallback_T_w_c):
        try:
            return self._cam_from_world_to_world_from_cam(image.cam_from_world())
        except Exception:
            return fallback_T_w_c.astype(np.float32)

    @staticmethod
    def _camera_calibration_matrix(camera, fallback_K):
        try:
            calib = camera.calibration_matrix
            K = calib() if callable(calib) else calib
            return np.asarray(K, dtype=np.float32)
        except Exception:
            return fallback_K.astype(np.float32)

    def _project_point_to_output_image(self, point3D_in_cam, camera, meta):
        x, y, z = np.asarray(point3D_in_cam, dtype=np.float64)
        if z <= 1e-3 or z > 1e3 or not np.isfinite(z):
            return None

        if self.use_input_intrinsics:
            K = meta["intrinsics"]
            u = round(x * K[0, 0] / z + K[0, 2])
            v = round(y * K[1, 1] / z + K[1, 2])
        else:
            xy = camera.img_from_cam(point3D_in_cam)
            if xy is None:
                return None
            u, v = np.rint(np.asarray(xy, dtype=np.float64)).astype(np.int64)

        if not (0 <= u < meta["width"] and 0 <= v < meta["height"]):
            return None
        return int(u), int(v), float(z)

    def _rasterize_depth(self, reconstruction, image_name, meta):
        depth = np.zeros((meta["height"], meta["width"]), dtype=np.float32)
        mask = np.zeros((meta["height"], meta["width"]), dtype=bool)
        fallback_K = np.asarray(meta["intrinsics"], dtype=np.float32)

        images_by_name = {
            image.name: image for image in reconstruction.images.values()
        }
        image = images_by_name.get(image_name)
        if image is None:
            return depth, mask, meta["T_w_c_input"].astype(np.float32), fallback_K

        camera = reconstruction.cameras[image.camera_id]
        cam_from_world = image.cam_from_world()
        for point2D in getattr(image, "points2D", []):
            if hasattr(point2D, "has_point3D") and not point2D.has_point3D():
                continue
            point3D_id = getattr(point2D, "point3D_id", None)
            if point3D_id is None or point3D_id < 0:
                continue
            if point3D_id not in reconstruction.points3D:
                continue

            point3D = reconstruction.points3D[point3D_id]
            point3D_in_cam = cam_from_world * point3D.xyz
            projection = self._project_point_to_output_image(point3D_in_cam, camera, meta)
            if projection is None:
                continue
            u, v, z = projection
            if not mask[v, u] or z < depth[v, u]:
                depth[v, u] = z
                mask[v, u] = True

        return (
            depth,
            mask,
            self._extract_image_pose(image, meta["T_w_c_input"]),
            self._camera_calibration_matrix(camera, fallback_K),
        )

    def _empty_sample_outputs(self, frames, batch_idx):
        outputs = []
        for frame_idx, frame in enumerate(frames):
            view = frame[0]
            image = self._image_to_uint8(view, batch_idx)
            outputs.append(
                {
                    "depth": np.zeros(image.shape[:2], dtype=np.float32),
                    "mask": np.zeros(image.shape[:2], dtype=bool),
                    "pose": self._tensor_to_numpy(view["T_w_c"], batch_idx).astype(np.float32),
                    "intrinsics": self._tensor_to_numpy(view["intrinsics"], batch_idx).astype(np.float32),
                }
            )
        return outputs

    def _prediction_to_colored_pointcloud(self, depth, mask, T_w_c, intrinsics, image):
        valid = mask & np.isfinite(depth) & (depth > 0)
        v, u = np.where(valid)
        if v.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )

        if self.pred_pointcloud_max_points is not None and v.size > self.pred_pointcloud_max_points:
            keep = np.linspace(0, v.size - 1, self.pred_pointcloud_max_points).astype(np.int64)
            v = v[keep]
            u = u[keep]

        z = depth[v, u].astype(np.float64)
        K = np.asarray(intrinsics, dtype=np.float64)
        x = (u.astype(np.float64) - K[0, 2]) * z / K[0, 0]
        y = (v.astype(np.float64) - K[1, 2]) * z / K[1, 1]
        points_cam = np.stack([x, y, z], axis=1)

        T_w_c = np.asarray(T_w_c, dtype=np.float64)
        points_world = points_cam @ T_w_c[:3, :3].T + T_w_c[:3, 3]
        colors = image[v, u].astype(np.float32) / 255.0
        return points_world.astype(np.float32), colors.astype(np.float32)

    def _write_colored_pointcloud(self, path, points, colors):
        import open3d as o3d

        path.parent.mkdir(parents=True, exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points))
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors))
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
        return pcd

    def visualize_prediction_pointclouds(self, frames, batch_idx, sample_outputs, output_dir):
        import open3d as o3d

        output_dir.mkdir(parents=True, exist_ok=True)
        merged_points = []
        merged_colors = []
        geometries = []

        for frame_idx, frame in enumerate(frames):
            view = frame[0]
            image = self._image_to_uint8(view, batch_idx)
            intrinsics = self._tensor_to_numpy(view["intrinsics"], batch_idx)
            pred = sample_outputs[frame_idx]
            points, colors = self._prediction_to_colored_pointcloud(
                pred["depth"],
                pred["mask"],
                pred["pose"],
                intrinsics,
                image,
            )

            frame_path = output_dir / f"frame_{frame_idx:06d}_points.ply"
            pcd = self._write_colored_pointcloud(frame_path, points, colors)
            geometries.append(pcd)
            if points.shape[0] > 0:
                merged_points.append(points)
                merged_colors.append(colors)

            depth_vis = np.zeros((*pred["depth"].shape, 3), dtype=np.uint8)
            if pred["mask"].any():
                valid_depth = pred["depth"][pred["mask"]]
                min_depth = np.percentile(valid_depth, 2)
                max_depth = np.percentile(valid_depth, 98)
                if max_depth <= min_depth:
                    max_depth = min_depth + 1e-6
                normalized = np.clip(
                    (pred["depth"] - min_depth) / (max_depth - min_depth), 0.0, 1.0
                )
                depth_vis = cv2.applyColorMap(
                    (255.0 * (1.0 - normalized)).astype(np.uint8),
                    cv2.COLORMAP_TURBO,
                )
                depth_vis[~pred["mask"]] = 0
            cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}_depth_vis.png"), depth_vis)
            cv2.imwrite(
                str(output_dir / f"frame_{frame_idx:06d}_image.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )

        if merged_points:
            merged_points = np.concatenate(merged_points, axis=0)
            merged_colors = np.concatenate(merged_colors, axis=0)
        else:
            merged_points = np.empty((0, 3), dtype=np.float32)
            merged_colors = np.empty((0, 3), dtype=np.float32)

        merged_pcd = self._write_colored_pointcloud(
            output_dir / "merged_pred_points.ply",
            merged_points,
            merged_colors,
        )
        if self.show_pred_pointcloud:
            o3d.visualization.draw_geometries([merged_pcd])

    def _run_sample(self, frames, batch_idx):
        sample_outputs = self._empty_sample_outputs(frames, batch_idx)

        with self._sample_work_dir() as tmpdir:
            image_dir = tmpdir / "images"
            outputs = tmpdir / "outputs"
            image_dir.mkdir(parents=True, exist_ok=True)
            outputs.mkdir(parents=True, exist_ok=True)
            poses_path = outputs / "input_poses.txt"

            image_list, metadata = self._prepare_sample(
                frames, batch_idx, image_dir, poses_path
            )
            rig_config = self._build_rig_config(frames, batch_idx)
            pose_priors = self._build_pose_priors_dict(frames, batch_idx, metadata)
            reconstruction = self._run_hloc(
                image_dir, outputs, image_list, poses_path, metadata,
                rig_config=rig_config, pose_priors=pose_priors,
                pose_prior_covariance=self.prior_position_covariance,
            )
            if reconstruction is None:
                return sample_outputs
            if reconstruction.num_reg_images() < self.min_model_size:
                return sample_outputs

            for frame_idx, _frame in enumerate(frames):
                image_name = self._view_name(frame_idx, 0)
                depth, mask, pose, intrinsics_pred = self._rasterize_depth(
                    reconstruction, image_name, metadata[image_name]
                )
                sample_outputs[frame_idx] = {
                    "depth": depth,
                    "mask": mask,
                    "pose": pose,
                    "intrinsics": intrinsics_pred,
                }

            if self.visualize_pred_pointcloud:
                pointcloud_dir = (
                    self.pred_pointcloud_dir / tmpdir.name
                    if self.pred_pointcloud_dir is not None
                    else outputs / "pred_pointcloud"
                )
                self.visualize_prediction_pointclouds(
                    frames,
                    batch_idx,
                    sample_outputs,
                    pointcloud_dir,
                )

        return sample_outputs

    def forward(self, frames):
        num_frame = len(frames)
        batch_size = frames[0][0]["undistorted_image"].shape[0]

        device = frames[0][0]["undistorted_image"].device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()

        batch_outputs = [self._run_sample(frames, batch_idx) for batch_idx in range(batch_size)]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime = time.time() - start

        results = []
        for frame_idx in range(num_frame):
            results.append(
                {
                    "pred_depth": np.stack(
                        [batch_outputs[b][frame_idx]["depth"] for b in range(batch_size)],
                        axis=0,
                    ),
                    "pred_depth_mask": np.stack(
                        [batch_outputs[b][frame_idx]["mask"] for b in range(batch_size)],
                        axis=0,
                    ),
                    "pred_depth_confidence": np.stack(
                        [
                            batch_outputs[b][frame_idx]["mask"].astype(np.float32)
                            for b in range(batch_size)
                        ],
                        axis=0,
                    ),
                    "pred_T_w_c": np.stack(
                        [batch_outputs[b][frame_idx]["pose"] for b in range(batch_size)],
                        axis=0,
                    ),
                    "pred_intrinsics": np.stack(
                        [batch_outputs[b][frame_idx]["intrinsics"] for b in range(batch_size)],
                        axis=0,
                    ),
                    "runtime": runtime / float(num_frame * max(batch_size, 1)),
                }
            )

        return results
