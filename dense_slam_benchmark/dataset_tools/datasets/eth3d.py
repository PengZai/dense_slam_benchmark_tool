import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dense_slam_benchmark.dataset_tools import utils

from .base import Dataset


class ETH3d(Dataset):
    def __init__(self, configs):
        self._configure_official_undistorted_cameras(configs)
        super().__init__(configs)
        self._build_samples_from_images_txt()

    def _get_undistorted_trajectory_path(self):
        data_source_idx = self.configs["system"]["use_data_source"]
        data_source = self.configs["data_source" + str(data_source_idx)]
        explicit_path = data_source.get("undistorted_trajectory_path")
        if explicit_path is not None:
            return explicit_path

        trajectory_path = data_source["trajectory_path"]
        candidate_path = trajectory_path
        replacements = [
            ("/multi_view_training_rig/", "/multi_view_training_rig_undistorted/"),
            ("/rig_calibration/", "/rig_calibration_undistorted/"),
            ("/multi_view_training_dslr_jpg/", "/multi_view_training_dslr_undistorted/"),
            ("/dslr_calibration_jpg/", "/dslr_calibration_undistorted/"),
        ]
        for src, dst in replacements:
            candidate_path = candidate_path.replace(src, dst)

        if candidate_path == trajectory_path:
            return None
        return candidate_path

    def _read_colmap_cameras_txt(self, cameras_txt_path):
        camera_models = {}
        with open(cameras_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(v) for v in parts[4:]]
                camera_models[camera_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params,
                }
        return camera_models

    def _configure_official_undistorted_cameras(self, configs):
        self.configs = configs
        undistorted_trajectory_path = self._get_undistorted_trajectory_path()
        if undistorted_trajectory_path is None:
            return

        undistorted_cameras_txt_path = str(Path(undistorted_trajectory_path).with_name("cameras.txt"))
        if not os.path.exists(undistorted_cameras_txt_path):
            return

        undistorted_camera_models = self._read_colmap_cameras_txt(undistorted_cameras_txt_path)
        data_source_idx = configs["system"]["use_data_source"]
        data_source = configs["data_source" + str(data_source_idx)]
        selected_camera_configs_by_id = {}

        def scale_intrinsics(intrinsics, src_resolution, dst_resolution):
            src_w, src_h = src_resolution
            dst_w, dst_h = dst_resolution
            sx = float(dst_w) / float(src_w)
            sy = float(dst_h) / float(src_h)
            fx, fy, cx, cy = intrinsics
            return [fx * sx, fy * sy, cx * sx, cy * sy]

        for used_camera_idx in data_source.get("used_camera_idxes", []):
            camera_config = configs["cameras"]["camera" + str(used_camera_idx)]
            camera_id = camera_config["id"]
            if camera_id not in undistorted_camera_models:
                raise ValueError(
                    f"ETH3d camera id {camera_id} is missing from official undistorted calibration: "
                    f"{undistorted_cameras_txt_path}"
                )
            undistorted_camera = undistorted_camera_models[camera_id]
            if undistorted_camera["model"] != "PINHOLE":
                raise ValueError(
                    f"Expected ETH3d undistorted camera {camera_id} to use PINHOLE, got "
                    f"{undistorted_camera['model']}"
                )
            fx, fy, cx, cy = undistorted_camera["params"][:4]
            official_resolution = [
                undistorted_camera["width"],
                undistorted_camera["height"],
            ]
            requested_resolution = camera_config.get("resolution", official_resolution)
            camera_config["undistorted_resolution"] = requested_resolution
            if requested_resolution != official_resolution:
                camera_config["undistorted_intrinsics"] = scale_intrinsics(
                    [fx, fy, cx, cy], official_resolution, requested_resolution
                )
            else:
                camera_config["undistorted_intrinsics"] = [fx, fy, cx, cy]
            selected_camera_configs_by_id[camera_id] = camera_config

        for used_sensor3d_idx in data_source.get("used_sensor3d_idxes", []):
            sensor3d_config = configs["sensor3ds"]["sensor3d" + str(used_sensor3d_idx)]
            camera_id = sensor3d_config.get("camera_id")
            if camera_id is None or camera_id not in undistorted_camera_models:
                continue
            aligned_camera_config = selected_camera_configs_by_id.get(camera_id)
            if aligned_camera_config is not None:
                sensor3d_config["undistorted_resolution"] = list(
                    aligned_camera_config["undistorted_resolution"]
                )
                sensor3d_config["undistorted_intrinsics"] = list(
                    aligned_camera_config["undistorted_intrinsics"]
                )
                continue

            undistorted_camera = undistorted_camera_models[camera_id]
            if undistorted_camera["model"] != "PINHOLE":
                continue
            fx, fy, cx, cy = undistorted_camera["params"][:4]
            official_resolution = [
                undistorted_camera["width"],
                undistorted_camera["height"],
            ]
            sensor3d_config["undistorted_resolution"] = official_resolution
            sensor3d_config["undistorted_intrinsics"] = [fx, fy, cx, cy]

    def _eth3d_extrinsics_to_T_w_cam(self, qw, qx, qy, qz, tx, ty, tz):
        """
        ETH3D / COLMAP images.txt stores the transform from world to camera.
        Convert it to T_w_cam for the current benchmark pipeline.
        """
        R_c_w = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        T_c_w = np.eye(4, dtype=np.float32)
        T_c_w[:3, :3] = R_c_w.astype(np.float32)
        T_c_w[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        return np.linalg.inv(T_c_w).astype(np.float32)
    
    def read_sensor3d_depth(self, sensor3d_data, sensor3d_name):
        sensor3d_path = os.path.join(sensor3d_data.config["sensor3dpath"], sensor3d_name)
        source_resolution = sensor3d_data.config.get(
            "original_resolution",
            sensor3d_data.config.get("resolution"),
        )
        width, height = source_resolution
        expected_num_values = width * height

        # ETH3D training depth maps are raw float32 dumps in row-major order.
        file_size = os.path.getsize(sensor3d_path)
        if file_size == expected_num_values * 4:
            depth_image = np.fromfile(sensor3d_path, dtype=np.float32)
            if depth_image.size != expected_num_values:
                raise ValueError(
                    f"Unexpected ETH3d depth size for {sensor3d_path}: "
                    f"expected {expected_num_values} float32 values, got {depth_image.size}"
                )
            return depth_image.reshape(height, width)

        return super().read_sensor3d_depth(sensor3d_data, sensor3d_name)

    def _parse_images_txt(self):
        image_records = []
        trajectory_path = self.data_source["trajectory_path"]
        with open(trajectory_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 10:
                    continue

                try:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_model_id = int(parts[8])
                except ValueError:
                    # ETH3D / COLMAP images.txt alternates image metadata lines with POINTS2D lines.
                    # Skip the keypoint-observation lines here.
                    continue
                name = parts[9]
                basename = os.path.basename(name)
                stem, suffix = os.path.splitext(basename)

                image_records.append(
                    {
                        "line_idx": line_idx,
                        "image_id": image_id,
                        "camera_model_id": camera_model_id,
                        "name": name,
                        "basename": basename,
                        "stem": stem,
                        "suffix": suffix,
                        "T_w_cam": self._eth3d_extrinsics_to_T_w_cam(qw, qx, qy, qz, tx, ty, tz),
                    }
                )
        return image_records

    def _record_matches_camera(self, record, camera_data):
        return record["camera_model_id"] == camera_data.id

    def _build_camera_record_index(self, image_records):
        records_by_camera = {camera_data.id: [] for camera_data in self.camera_data_lists}

        for record in image_records:
            matched_camera_ids = [
                camera_data.id
                for camera_data in self.camera_data_lists
                if self._record_matches_camera(record, camera_data)
            ]

            if len(matched_camera_ids) == 1:
                records_by_camera[matched_camera_ids[0]].append(record)
                continue

            if len(matched_camera_ids) > 1:
                raise ValueError(
                    f"ETH3d image record {record['name']} with CAMERA_ID={record['camera_model_id']} "
                    "matched more than one configured camera. camera.id values must uniquely match "
                    "images.txt CAMERA_ID."
                )

        for camera_id in records_by_camera:
            records_by_camera[camera_id].sort(key=lambda rec: (rec["image_id"], rec["basename"]))

        lookup_by_camera = {
            camera_id: {record["stem"]: idx for idx, record in enumerate(records)}
            for camera_id, records in records_by_camera.items()
        }
        return records_by_camera, lookup_by_camera

    def _build_stem_to_camera_records(self, image_records):
        stem_to_camera_records = {}
        for record in image_records:
            matched_camera_ids = [
                camera_data.id
                for camera_data in self.camera_data_lists
                if self._record_matches_camera(record, camera_data)
            ]

            if len(matched_camera_ids) == 0:
                continue

            if len(matched_camera_ids) > 1:
                raise ValueError(
                    f"ETH3d image record {record['name']} with CAMERA_ID={record['camera_model_id']} "
                    "matched more than one configured camera. camera.id values must uniquely match "
                    "images.txt CAMERA_ID."
                )

            stem_to_camera_records.setdefault(record["stem"], {})
            for camera_id in matched_camera_ids:
                stem_to_camera_records[record["stem"]].setdefault(camera_id, []).append(record)

        for camera_records in stem_to_camera_records.values():
            for camera_id in camera_records:
                camera_records[camera_id].sort(key=lambda rec: (rec["image_id"], rec["basename"]))

        return stem_to_camera_records

    def _build_sensor_name_lookup(self):
        sensor_name_lookup = {}
        for sensor3d_data in self.sensor3d_data_list:
            sensor_name_lookup[sensor3d_data.id] = {
                os.path.splitext(sensor_name)[0]: sensor_name
                for sensor_name in sensor3d_data.sensor3d_names
            }
        return sensor_name_lookup

    def _build_sample_from_reference_record(
        self,
        sample_idx,
        sample_stem,
        sample_camera_records,
        records_by_camera,
        sensor_name_lookup,
    ):
        synchronized_image_data_list = []
        camera_records_for_sample = {}

        for camera_data in self.camera_data_lists:
            camera_id = camera_data.id
            candidate_records = sample_camera_records.get(camera_id, [])
            if len(candidate_records) == 0:
                return None

            camera_record = candidate_records[0]
            camera_record_idx = next(
                (
                    idx
                    for idx, record in enumerate(records_by_camera[camera_id])
                    if record["image_id"] == camera_record["image_id"]
                ),
                None,
            )
            if camera_record_idx is None:
                return None

            camera_records_for_sample[camera_id] = {
                "record": camera_record,
                "record_idx": camera_record_idx,
            }
            synchronized_image_data_list.append(
                {
                    "name": camera_record["basename"],
                    "ts": sample_stem,
                    "ts_diff_with_sample_ts": 0,
                    "camera_id": camera_id,
                    "T_w_cam_idx": camera_record["T_w_cam"],
                }
            )

        synchronized_sensor3d_data_list_list = []
        for sensor3d_idx_in_use, sensor3d_data in enumerate(self.sensor3d_data_list):
            sensor3d_config = sensor3d_data.config
            camera_id = sensor3d_config["camera_id"]
            if camera_id not in camera_records_for_sample:
                return None

            aligned_camera_record = camera_records_for_sample[camera_id]
            aligned_camera_idx = aligned_camera_record["record_idx"]
            aligned_camera_records = records_by_camera[camera_id]
            num_cumulation = self.data_source["num_cumulation"][sensor3d_idx_in_use]

            lower_idx = max(0, aligned_camera_idx - num_cumulation)
            upper_idx = min(len(aligned_camera_records), aligned_camera_idx + num_cumulation + 1)

            synchronized_sensor3d_data_list = []
            for idx in range(lower_idx, upper_idx):
                camera_record = aligned_camera_records[idx]
                sensor_name = sensor_name_lookup[sensor3d_data.id].get(camera_record["stem"])
                if sensor_name is None:
                    continue

                synchronized_sensor3d_data_list.append(
                    {
                        "name": sensor_name,
                        "ts_diff_with_sample_ts": abs(idx - aligned_camera_idx),
                        "T_w_sensor3d": camera_record["T_w_cam"],
                    }
                )

            if len(synchronized_sensor3d_data_list) == 0:
                return None

            synchronized_sensor3d_data_list_list.append(synchronized_sensor3d_data_list)

        sample = {
            "id": sample_idx,
            "ts": sample_stem,
            "str_ts": sample_stem,
            "T_w_p": camera_records_for_sample[self.camera_data_lists[0].id]["record"]["T_w_cam"],
            "synchronized_image_data_list": synchronized_image_data_list,
            "synchronized_sensor3d_data_list_list": synchronized_sensor3d_data_list_list,
        }
        return sample

    def _build_samples_from_images_txt(self):
        image_records = self._parse_images_txt()
        records_by_camera, _ = self._build_camera_record_index(image_records)
        stem_to_camera_records = self._build_stem_to_camera_records(image_records)
        sensor_name_lookup = self._build_sensor_name_lookup()

        reference_camera_id = self.camera_data_lists[0].id
        built_samples = []
        valid_sample_stems = sorted(
            stem
            for stem, camera_records in stem_to_camera_records.items()
            if reference_camera_id in camera_records and len(camera_records[reference_camera_id]) > 0
        )

        for sample_stem in valid_sample_stems:
            sample = self._build_sample_from_reference_record(
                sample_idx=len(built_samples),
                sample_stem=sample_stem,
                sample_camera_records=stem_to_camera_records[sample_stem],
                records_by_camera=records_by_camera,
                sensor_name_lookup=sensor_name_lookup,
            )
            if sample is not None:
                built_samples.append(sample)

        if len(built_samples) == 0:
            raise ValueError(
                "Failed to build ETH3d samples from images.txt. Please check that the configured "
                "used_camera_idxes / used_sensor3d_idxes match the image and sensor file names."
            )

        self.samples = built_samples
