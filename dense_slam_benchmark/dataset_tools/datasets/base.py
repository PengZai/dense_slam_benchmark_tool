import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from pytransform3d.transformations import transform_sclerp

from dense_slam_benchmark.dataset_tools import utils


class Sensor3dData:
    def __init__(self, sensor3d_config):
        super().__init__()
        self.config = sensor3d_config
        self.id = sensor3d_config["id"]
        self.sensor3d_names = sorted(os.listdir(self.config["sensor3dpath"]))


class PointCloud(Sensor3dData):
    def __init__(self, sensor3d_config):
        super().__init__(sensor3d_config)


class PointCloudInCameraCoordinate(Sensor3dData):
    def __init__(self, sensor3d_config):
        super().__init__(sensor3d_config)


class ImageDepth(Sensor3dData):
    def __init__(self, sensor3d_config):
        super().__init__(sensor3d_config)
        intrinsics = sensor3d_config.get("intrinsics", sensor3d_config.get("original_intrinsics"))
        if intrinsics is None:
            raise ValueError(
                f"sensor3d{sensor3d_config['id']} is missing intrinsics/original_intrinsics"
            )
        source_resolution = sensor3d_config.get("original_resolution")
        if source_resolution is None:
            source_resolution = sensor3d_config.get("resolution")
        target_resolution = sensor3d_config.get("undistorted_resolution")
        if target_resolution is None:
            target_resolution = sensor3d_config.get("resolution", source_resolution)
        self.K, self.remap1, self.remap2 = utils.calculateUndistortedRemap(
            sensor3d_config["distortion_model"],
            source_resolution,
            intrinsics,
            sensor3d_config["distortion_coeffs"],
            target_resolution=target_resolution,
            target_intrinsics=sensor3d_config.get("undistorted_intrinsics"),
        )


class CameraData:
    def __init__(self, camera_config):
        super().__init__()
        self.config = camera_config
        self.id = camera_config["id"]
        self.image_names = sorted(os.listdir(self.config["imagepath"]))
        source_resolution = camera_config.get("original_resolution")
        if source_resolution is None:
            source_resolution = camera_config.get("resolution")
        target_resolution = camera_config.get("undistorted_resolution")
        if target_resolution is None:
            target_resolution = camera_config.get("resolution", source_resolution)

        self.K, self.remap1, self.remap2 = utils.calculateUndistortedRemap(
            camera_config["distortion_model"],
            source_resolution,
            camera_config["original_intrinsics"],
            camera_config["distortion_coeffs"],
            target_resolution=target_resolution,
            target_intrinsics=camera_config.get("undistorted_intrinsics"),
        )

        camera_config["undistorted_intrinsics"] = [
            float(self.K[0, 0]),
            float(self.K[1, 1]),
            float(self.K[0, 2]),
            float(self.K[1, 2]),
        ]
        if "undistorted_resolution" not in camera_config:
            camera_config["undistorted_resolution"] = [int(self.remap1.shape[1]), int(self.remap1.shape[0])]


class Dataset:
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.sensor3d_data_list = []
        self.camera_data_lists = []
        self.samples = []

        data_source_idx = configs["system"]["use_data_source"]
        self.data_source = configs["data_source" + str(data_source_idx)]
        self.validate_used_sensor3d_idxes()

        if "used_sensor3d_idxes" in self.data_source:
            for used_sensord3d_id in self.data_source["used_sensor3d_idxes"]:
                sensor3d_config_i = configs["sensor3ds"]["sensor3d" + str(used_sensord3d_id)]
                if sensor3d_config_i["sensor3dtype"] in {"pointcloud", "pointcloud_in_camera_coordinate"}:
                    self.sensor3d_data_list.append(PointCloud(sensor3d_config_i))
                elif sensor3d_config_i["sensor3dtype"] == "imagedepth":
                    self.sensor3d_data_list.append(ImageDepth(sensor3d_config_i))
                else:
                    raise ValueError(
                        f"Unsupported sensor3dtype: {sensor3d_config_i['sensor3dtype']}"
                    )

        if "used_camera_idxes" in self.data_source:
            for used_camera_id in self.data_source["used_camera_idxes"]:
                camera_config_i = configs["cameras"]["camera" + str(used_camera_id)]
                self.camera_data_lists.append(CameraData(camera_config_i))

        self.make_output_directories()

    def get_start_idx(self):
        return int(self.configs["system"]["start_idx"])

    def get_end_idx_exclusive(self, total_count=None):
        if total_count is None:
            total_count = len(self.samples)
        end_idx = int(self.configs["system"]["end_idx"])
        if end_idx < 0:
            return total_count
        return min(end_idx + 1, total_count)

    def is_sample_idx_selected(self, sample_idx, total_count=None):
        if total_count is None:
            total_count = len(self.samples)
        start_idx = self.get_start_idx()
        end_idx_exclusive = self.get_end_idx_exclusive(total_count)
        return start_idx <= sample_idx < end_idx_exclusive

    def get_selected_samples(self):
        start_idx = self.get_start_idx()
        end_idx_exclusive = self.get_end_idx_exclusive(len(self.samples))
        return self.samples[start_idx:end_idx_exclusive]

    def read_sensor3d_depth(self, sensor3d_data, sensor3d_name):
        sensor3d_path = os.path.join(sensor3d_data.config["sensor3dpath"], sensor3d_name)
        suffix = Path(sensor3d_name).suffix.lower()
        if suffix in {".tiff", ".tif", ".png", ".jpg", ".jpeg"}:
            depth_image = cv2.imread(sensor3d_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise ValueError(f"Failed to read depth image: {sensor3d_path}")
            return depth_image
        if suffix == ".npy":
            return np.load(sensor3d_path)
        raise ValueError(f"Unsupported depth file format for sensor3d '{sensor3d_name}'")

    def load_sensor3d_points_h(self, sensor3d_data, sensor3d_name):
        sensor3d_config_i = sensor3d_data.config
        suffix = Path(sensor3d_name).suffix.lower()

        if sensor3d_config_i["sensor3dtype"] in {"pointcloud", "pointcloud_in_camera_coordinate"}:
            if suffix != ".pcd":
                raise ValueError(
                    f"Unsupported point cloud format for sensor3d '{sensor3d_name}'"
                )
            pcd = o3d.io.read_point_cloud(os.path.join(sensor3d_config_i["sensor3dpath"], sensor3d_name))
            if sensor3d_config_i["down_sample_rate"] < 1.0:
                pcd = pcd.random_down_sample(sensor3d_config_i["down_sample_rate"])
            points_sensor3d = np.asarray(pcd.points, dtype="f4")
            return np.hstack(
                [points_sensor3d, np.ones((points_sensor3d.shape[0], 1), dtype=np.float32)]
            )

        if sensor3d_config_i["sensor3dtype"] == "imagedepth":
            depth_image = self.read_sensor3d_depth(sensor3d_data, sensor3d_name)
            K = sensor3d_data.K
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            distortion_coeffs = np.asarray(
                sensor3d_data.config["distortion_coeffs"], dtype=np.float32
            ).reshape(-1)
            if np.any(np.abs(distortion_coeffs) > 1e-12):
                depth_image = cv2.remap(
                    depth_image, sensor3d_data.remap1, sensor3d_data.remap2, cv2.INTER_NEAREST
                )

            v, u = np.indices(depth_image.shape, dtype="f4")
            mask = np.isfinite(depth_image) & (depth_image > 1e-3)
            x = (u[mask] - cx) * depth_image[mask] / fx
            y = (v[mask] - cy) * depth_image[mask] / fy
            z = depth_image[mask]
            ones = np.ones_like(z, dtype="f4")
            points_sensor3d_h = np.stack((x, y, z, ones), axis=1)
            if sensor3d_config_i["down_sample_rate"] < 1.0:
                points_sensor3d_h = utils.voxel_downsample_np(
                    points_sensor3d_h, sensor3d_config_i["down_sample_rate"]
                )
            return points_sensor3d_h

        raise ValueError(f"Unsupported sensor3dtype: {sensor3d_config_i['sensor3dtype']}")

    def validate_used_sensor3d_idxes(self):
        if "used_sensor3d_idxes" not in self.data_source:
            return

        trajectory_type = self.data_source.get("trajectorytype")
        if trajectory_type != "poses_for_all_cameras":
            return

        allowed_sensor_types = {"imagedepth", "pointcloud_in_camera_coordinate"}
        invalid_sensor_entries = []

        for used_sensor3d_idx in self.data_source["used_sensor3d_idxes"]:
            sensor3d_config = self.configs["sensor3ds"]["sensor3d" + str(used_sensor3d_idx)]
            sensor3dtype = sensor3d_config["sensor3dtype"]
            if sensor3dtype not in allowed_sensor_types:
                invalid_sensor_entries.append(
                    f"sensor3d{used_sensor3d_idx} ({sensor3dtype})"
                )
                continue

            if "camera_id" not in sensor3d_config:
                invalid_sensor_entries.append(
                    f"sensor3d{used_sensor3d_idx} ({sensor3dtype}, missing camera_id)"
                )

        if invalid_sensor_entries:
            raise ValueError(
                "For trajectorytype='poses_for_all_cameras', used_sensor3d_idxes may only "
                "contain camera-aligned sensors of type 'imagedepth' or "
                "'pointcloud_in_camera_coordinate' with a valid camera_id. Invalid entries: "
                + ", ".join(invalid_sensor_entries)
            )

    def filter_depth_by_knn(
        self,
        cumulated_u_for_depthimage,
        cumulated_v_for_depthimage,
        cumulated_z_for_depthimage,
        k=3,
        base_tol=0.02,
        mad_scale=2.5,
    ):
        """
        Remove isolated depth outliers by checking each projected sample against
        the depths of its nearest neighbors in image space.

        The input arrays usually come from many 3D samples being projected onto
        the same image. Small pose noise, reprojection noise, or mixed surfaces
        can create pixels whose depth is inconsistent with nearby projections.
        This filter keeps a sample only if its depth is close to the local depth
        consensus formed by its K nearest neighbors in the `(u, v)` image plane.

        The tolerance is adaptive:
        - `base_tol` provides a minimum allowed depth difference.
        - `mad_scale * 1.4826 * MAD` enlarges the tolerance in locally noisy
          regions, where MAD is the median absolute deviation of neighbor depths.

        This makes the filter strict in smooth regions while avoiding over-
        pruning near depth discontinuities or in noisier measurements.
        """
        u = np.asarray(cumulated_u_for_depthimage)
        v = np.asarray(cumulated_v_for_depthimage)
        z = np.asarray(cumulated_z_for_depthimage)

        if not (len(u) == len(v) == len(z)):
            raise ValueError("u, v, z must have the same length")
        if len(u) <= 1:
            return u.copy(), v.copy(), z.copy()

        uv = np.column_stack((u, v)).astype(np.float32)
        zf = z.astype(np.float32)
        tree = cKDTree(uv)

        # Query k+1 neighbors because the closest point is the sample itself.
        k_eff = min(k + 1, len(uv))
        dist, idx = tree.query(uv, k=k_eff, workers=-1)

        if idx.ndim == 1:
            idx = idx[:, None]
            dist = dist[:, None]

        # Drop the self-neighbor and keep only spatial neighbors.
        nn_idx = idx[:, 1:]
        if nn_idx.shape[1] == 0:
            return u.copy(), v.copy(), z.copy()

        neighbor_z = zf[nn_idx]

        # Use a robust local depth model: the median is the expected depth and
        # the median absolute deviation estimates how spread the neighborhood is.
        median_z = np.median(neighbor_z, axis=1)
        mad = np.median(np.abs(neighbor_z - median_z[:, None]), axis=1)
        adaptive_tol = np.maximum(base_tol, mad_scale * 1.4826 * mad)
        keep_mask = np.abs(zf - median_z) <= adaptive_tol

        return u[keep_mask], v[keep_mask], z[keep_mask]

    def make_output_directories(self):
        output_dir = self.configs["output"]["path"]
        os.makedirs(output_dir, exist_ok=True)

        data_source_idx = self.configs["system"]["use_data_source"]
        data_source = self.configs["data_source" + str(data_source_idx)]

        save_config = {
            "system": self.configs["system"],
            "cameras": self.configs["cameras"],
            "sensor3ds": self.configs["sensor3ds"],
            "data_source": data_source,
        }

        output_scene_dict = {}
        for used_camera_idx in data_source["used_camera_idxes"]:
            output_relative_path_dict = {}
            output_relative_path_dict["sparse_depth_relative_path"] = " "
            output_relative_path_dict["sparse_pointcloud_relative_path"] = " "

            camera_config_i = self.configs["cameras"]["camera" + str(used_camera_idx)]
            output_image_root_dir = camera_config_i["name"]
            os.makedirs(os.path.join(output_dir, output_image_root_dir), exist_ok=True)

            output_relative_path_dict["undistorted_images_path"] = os.path.join(
                output_image_root_dir, "undistorted_images"
            )
            os.makedirs(
                os.path.join(output_dir, output_relative_path_dict["undistorted_images_path"]),
                exist_ok=True,
            )

            name_for_GT_dir = ""
            for i, used_sensor3d_idx in enumerate(data_source["used_sensor3d_idxes"]):
                sensor3d_config_i = self.configs["sensor3ds"]["sensor3d" + str(used_sensor3d_idx)]
                name_for_GT_dir += sensor3d_config_i["name"] + "_" + "c" + str(
                    data_source["num_cumulation"][i]
                )
                if i < len(data_source["used_sensor3d_idxes"]) - 1:
                    name_for_GT_dir += "_"

            output_relative_path_dict["name_for_GT_dir"] = name_for_GT_dir
            output_relative_path_dict["GT_pose_output_relative_path"] = os.path.join(
                output_image_root_dir, name_for_GT_dir
            )
            os.makedirs(
                os.path.join(output_dir, output_relative_path_dict["GT_pose_output_relative_path"]),
                exist_ok=True,
            )
            with open(
                os.path.join(
                    output_dir,
                    output_relative_path_dict["GT_pose_output_relative_path"],
                    "config.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(save_config, f, ensure_ascii=False, indent=2)

            with open(
                os.path.join(
                    output_dir,
                    output_relative_path_dict["GT_pose_output_relative_path"],
                    "Twc.txt",
                ),
                "w",
            ) as f:
                f.write("#timestamp/index x y z q_x q_y q_z q_w\n")

            output_relative_path_dict["GT_depth_output_relative_path"] = os.path.join(
                output_image_root_dir, name_for_GT_dir, "depth"
            )
            os.makedirs(
                os.path.join(output_dir, output_relative_path_dict["GT_depth_output_relative_path"]),
                exist_ok=True,
            )
            output_relative_path_dict["GT_depth_vis_output_relative_path"] = os.path.join(
                output_image_root_dir, name_for_GT_dir, "depth_vis"
            )
            os.makedirs(
                os.path.join(
                    output_dir, output_relative_path_dict["GT_depth_vis_output_relative_path"]
                ),
                exist_ok=True,
            )
            output_relative_path_dict["GT_pointcloud_output_relative_path"] = os.path.join(
                output_image_root_dir, name_for_GT_dir, "pointcloud"
            )
            os.makedirs(
                os.path.join(
                    output_dir, output_relative_path_dict["GT_pointcloud_output_relative_path"]
                ),
                exist_ok=True,
            )

            self.configs["cameras"]["camera" + str(used_camera_idx)][
                "output_relative_path_dict"
            ] = output_relative_path_dict
            output_scene_dict["camera" + str(used_camera_idx)] = self.configs["cameras"][
                "camera" + str(used_camera_idx)
            ]

        if not os.path.exists(os.path.join(output_dir, "scene.json")):
            with open(os.path.join(output_dir, "scene.json"), "w", encoding="utf-8") as f:
                json.dump(output_scene_dict, f, ensure_ascii=False, indent=2)

    def readDatasample(self):
        idx = 0
        with open(self.data_source["trajectory_path"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                str_ts = line.split()[0]
                ts, x, y, z, qx, qy, qz, qw = map(float, line.split())
                T_w_p = utils.pose_to_T(x, y, z, qx, qy, qz, qw)
                datasample = {"id": idx, "ts": ts, "str_ts": str_ts, "T_w_p": T_w_p}
                self.samples.append(datasample)
                idx += 1

    def getPoses(self):
        pose_list = []
        for datasample in self.samples:
            pose_list.append(datasample["T_w_p"])
        return np.stack(pose_list, dtype="f4")

    def getSynchronizedPose(self, sensor_ts):
        T_w_p = None
        isSync, synchronized_pose_idx = utils.getSynchronizedPoseIdx(sensor_ts, self.samples, 1e-6)
        if isSync:
            T_w_p = self.samples[synchronized_pose_idx]["T_w_p"]
        else:
            closet_ts = self.samples[synchronized_pose_idx]["ts"]
            if sensor_ts > closet_ts and synchronized_pose_idx + 1 < len(self.samples):
                ts_start = closet_ts
                T_start = self.samples[synchronized_pose_idx]["T_w_p"]
                ts_end = self.samples[synchronized_pose_idx + 1]["ts"]
                T_end = self.samples[synchronized_pose_idx + 1]["T_w_p"]
            elif sensor_ts <= closet_ts and synchronized_pose_idx - 1 >= 0:
                ts_start = self.samples[synchronized_pose_idx - 1]["ts"]
                T_start = self.samples[synchronized_pose_idx - 1]["T_w_p"]
                ts_end = closet_ts
                T_end = self.samples[synchronized_pose_idx]["T_w_p"]
            else:
                return None

            t = (sensor_ts - ts_start) / (ts_end - ts_start)
            T_w_p = transform_sclerp(T_start, T_end, t)

        return T_w_p

    def loadSyncrhonizedData(self, sample):
        pose_idx = sample["id"]
        synchronized_image_data_list = []

        for i, camera_data in enumerate(self.camera_data_lists):
            T_pose_cam_idx = np.array(self.data_source["T_pose_used_cam_idx" + str(i)], dtype="f4")
            image_name_closest_with_sample_ts = camera_data.image_names[pose_idx]
            T_w_p_syncrhonize_with_image_ts = sample["T_w_p"]
            T_w_cam_idx_syncrhonize_with_image_ts = T_w_p_syncrhonize_with_image_ts @ T_pose_cam_idx
            synchronized_image_data_list.append(
                {
                    "name": image_name_closest_with_sample_ts,
                    "ts": str(pose_idx),
                    "ts_diff_with_sample_ts": 0,
                    "camera_id": camera_data.id,
                    "T_w_cam_idx": T_w_cam_idx_syncrhonize_with_image_ts,
                }
            )

        sample["synchronized_image_data_list"] = synchronized_image_data_list

        synchronized_sensor3d_data_list_list = []
        for idx, sensor3d_data in enumerate(self.sensor3d_data_list):
            synchronized_sensor3d_data_list = []
            idx_closest_with_sample_ts = pose_idx
            sensor3d_name_closest_with_sample_ts = sensor3d_data.sensor3d_names[
                idx_closest_with_sample_ts
            ]
            T_w_p_syncrhonize_with_sensor3d_ts = sample["T_w_p"]
            T_p_sensor3d = np.array(
                self.data_source["T_pose_used_sensor3d_idx" + str(self.data_source["used_sensor3d_idxes"][idx])]
            )
            T_w_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p_syncrhonize_with_sensor3d_ts @ T_p_sensor3d

            sample["synchronized_sensor3d_data"] = {
                "name": sensor3d_name_closest_with_sample_ts,
                "ts_diff_with_sample_ts": 0,
                "T_w_sensor3d": T_w_sensor3d_syncrhonize_with_sensor3d_ts,
            }

            if self.data_source["num_cumulation"][idx] > 0:
                upper_sensor3d_idx = idx_closest_with_sample_ts + self.data_source["num_cumulation"][idx] + 1
                if upper_sensor3d_idx > len(sensor3d_data.sensor3d_names) - 1:
                    upper_sensor3d_idx = len(sensor3d_data.sensor3d_names) - 1

                lower_sensor3d_idx = idx_closest_with_sample_ts - self.data_source["num_cumulation"][idx]
                if lower_sensor3d_idx < 0:
                    lower_sensor3d_idx = 0

                if upper_sensor3d_idx > lower_sensor3d_idx:
                    cumulated_sensor3d_name_list = sensor3d_data.sensor3d_names[
                        lower_sensor3d_idx:upper_sensor3d_idx
                    ]
                    cumulated_samples = self.samples[lower_sensor3d_idx:upper_sensor3d_idx]

                    for i, sensor3d_name in enumerate(cumulated_sensor3d_name_list):
                        T_w_p = cumulated_samples[i]["T_w_p"]
                        T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p @ T_p_sensor3d
                        synchronized_sensor3d_data_list.append(
                            {
                                "name": sensor3d_name,
                                "ts_diff_with_sample_ts": 0,
                                "T_w_sensor3d": T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts,
                            }
                        )
            else:
                synchronized_sensor3d_data_list.append(
                    {
                        "name": sensor3d_name_closest_with_sample_ts,
                        "ts_diff_with_sample_ts": 0,
                        "T_w_sensor3d": T_w_sensor3d_syncrhonize_with_sensor3d_ts,
                    }
                )

            synchronized_sensor3d_data_list_list.append(synchronized_sensor3d_data_list)

        sample["synchronized_sensor3d_data_list_list"] = synchronized_sensor3d_data_list_list

    def loadAsyncrhonizedData(self, sample):
        synchronized_image_data_list = []

        for i, camera_data in enumerate(self.camera_data_lists):
            T_pose_cam_idx = np.array(self.data_source["T_pose_used_cam_idx" + str(i)], dtype="f4")
            image_idx_closest_with_sample_ts = utils.getSensorIdxWithClosestTimeStamp(
                sample["ts"], camera_data.image_names
            )
            if image_idx_closest_with_sample_ts == -1:
                return None

            image_name_closest_with_sample_ts = camera_data.image_names[image_idx_closest_with_sample_ts]
            image_ts = utils.timestamp_str_to_float(image_name_closest_with_sample_ts.split(".")[0])
            T_w_p_syncrhonize_with_image_ts = self.getSynchronizedPose(image_ts)
            if T_w_p_syncrhonize_with_image_ts is None:
                return None

            T_w_cam_idx_syncrhonize_with_image_ts = T_w_p_syncrhonize_with_image_ts @ T_pose_cam_idx
            synchronized_image_data_list.append(
                {
                    "name": image_name_closest_with_sample_ts,
                    "ts": image_ts,
                    "ts_diff_with_sample_ts": abs(sample["ts"] - image_ts),
                    "camera_id": camera_data.id,
                    "T_w_cam_idx": T_w_cam_idx_syncrhonize_with_image_ts,
                }
            )

        sample["synchronized_image_data_list"] = synchronized_image_data_list

        synchronized_sensor3d_data_list_list = []
        for idx, sensor3d_data in enumerate(self.sensor3d_data_list):
            idx_closest_with_sample_ts = utils.getSensorIdxWithClosestTimeStamp(
                sample["ts"], sensor3d_data.sensor3d_names
            )
            if idx_closest_with_sample_ts == -1:
                return None
            synchronized_sensor3d_data_list = []
            T_p_sensor3d = np.array(
                self.data_source["T_pose_used_sensor3d_idx" + str(self.data_source["used_sensor3d_idxes"][idx])]
            )

            if self.data_source["num_cumulation"][idx] > 0:
                upper_sensor3d_idx = idx_closest_with_sample_ts + self.data_source["num_cumulation"][idx] + 1
                if upper_sensor3d_idx > len(sensor3d_data.sensor3d_names) - 1:
                    upper_sensor3d_idx = len(sensor3d_data.sensor3d_names) - 1

                lower_sensor3d_idx = idx_closest_with_sample_ts - self.data_source["num_cumulation"][idx]
                if lower_sensor3d_idx < 0:
                    lower_sensor3d_idx = 0

                if upper_sensor3d_idx > lower_sensor3d_idx:
                    cumulated_sensor3d_name_list = sensor3d_data.sensor3d_names[
                        lower_sensor3d_idx:upper_sensor3d_idx
                    ]
                    for sensor3d_name in cumulated_sensor3d_name_list:
                        sensor3d_ts = utils.timestamp_str_to_float(sensor3d_name.split(".")[0])
                        T_w_p = self.getSynchronizedPose(sensor3d_ts)
                        if T_w_p is None:
                            continue
                        T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p @ T_p_sensor3d
                        synchronized_sensor3d_data_list.append(
                            {
                                "name": sensor3d_name,
                                "ts_diff_with_sample_ts": abs(sample["ts"] - sensor3d_ts),
                                "T_w_sensor3d": T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts,
                            }
                        )
            else:
                sensor3d_name_closest_with_sample_ts = sensor3d_data.sensor3d_names[
                    idx_closest_with_sample_ts
                ]
                sensor3d_ts = utils.timestamp_str_to_float(sensor3d_name_closest_with_sample_ts.split(".")[0])
                T_w_p_syncrhonize_with_sensor3d_ts = self.getSynchronizedPose(sensor3d_ts)
                if T_w_p_syncrhonize_with_sensor3d_ts is None:
                    return None

                T_w_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p_syncrhonize_with_sensor3d_ts @ T_p_sensor3d
                synchronized_sensor3d_data_list.append(
                    {
                        "name": sensor3d_name_closest_with_sample_ts,
                        "ts_diff_with_sample_ts": abs(sample["ts"] - sensor3d_ts),
                        "T_w_sensor3d": T_w_sensor3d_syncrhonize_with_sensor3d_ts,
                    }
                )

            synchronized_sensor3d_data_list_list.append(synchronized_sensor3d_data_list)

        sample["synchronized_sensor3d_data_list_list"] = synchronized_sensor3d_data_list_list

    def loadBenchmarkData(self, sample):
        cumulated_points_w_h_list = []
        for sensor3d_i, synchronized_sensor3d_data_list in enumerate(
            sample["synchronized_sensor3d_data_list_list"]
        ):
            sensor3d_data = self.sensor3d_data_list[sensor3d_i]
            sensor3d_config_i = self.sensor3d_data_list[sensor3d_i].config

            for synchronized_sensor3d_data in synchronized_sensor3d_data_list:
                sensor3d_name = synchronized_sensor3d_data["name"]
                T_w_sensor3d = synchronized_sensor3d_data["T_w_sensor3d"]
                scale_transformation_matrix = np.array(
                    [
                        [sensor3d_config_i["xyz_scale_factor"][0], 0, 0, 0],
                        [0, sensor3d_config_i["xyz_scale_factor"][1], 0, 0],
                        [0, 0, sensor3d_config_i["xyz_scale_factor"][2], 0],
                        [0, 0, 0, 1],
                    ],
                    dtype="f4",
                )
                points_sensor3d_h = self.load_sensor3d_points_h(sensor3d_data, sensor3d_name)

                points_w_h = (T_w_sensor3d @ points_sensor3d_h.T).T
                cumulated_points_w_h_list.append(points_w_h)

        cumulated_points_w_h = np.vstack(cumulated_points_w_h_list, dtype="f4")

        maximum_depth_for_depthimage = self.configs["system"]["maximum_depth_for_depthimage"]
        minimum_depth_for_depthimage = self.configs["system"]["minimum_depth_for_depthimage"]
        maximum_z_for_pointcloud = self.configs["system"]["maximum_z_for_pointcloud"]
        minimum_z_for_pointcloud = self.configs["system"]["minimum_z_for_pointcloud"]

        for camera_data_idx, synchronized_image_data in enumerate(sample["synchronized_image_data_list"]):
            image_name = synchronized_image_data["name"]
            image = cv2.imread(os.path.join(self.camera_data_lists[camera_data_idx].config["imagepath"], image_name))
            K = self.camera_data_lists[camera_data_idx].K

            undistorted_image = cv2.remap(
                image,
                self.camera_data_lists[camera_data_idx].remap1,
                self.camera_data_lists[camera_data_idx].remap2,
                cv2.INTER_LINEAR,
            )
            height, width = undistorted_image.shape[0], undistorted_image.shape[1]

            cumulated_sensor3d_depth = np.zeros(
                (undistorted_image.shape[0], undistorted_image.shape[1]), dtype="f4"
            )

            cumulated_p_c_h = (
                scale_transformation_matrix
                @ np.linalg.inv(synchronized_image_data["T_w_cam_idx"])
                @ cumulated_points_w_h.T
            ).T
            cumulated_p_c = cumulated_p_c_h[:, :3]
            K_T = K.T
            cumulated_z = cumulated_p_c[:, 2]
            cumulated_p_c_in_norm_plane = cumulated_p_c @ K_T
            cumulated_uv1 = (cumulated_p_c_in_norm_plane / cumulated_p_c_in_norm_plane[:, -1:]).round().astype(np.int64)
            cumulated_u = cumulated_uv1[:, 0]
            cumulated_v = cumulated_uv1[:, 1]

            mask_for_depthimage = (
                (cumulated_z > minimum_depth_for_depthimage)
                & (cumulated_z <= maximum_depth_for_depthimage)
                & (cumulated_u >= 0)
                & (cumulated_u < width)
                & (cumulated_v >= 0)
                & (cumulated_v < height)
            )
            mask_for_pointcloud = (
                (cumulated_z > minimum_z_for_pointcloud)
                & (cumulated_z <= maximum_z_for_pointcloud)
                & (cumulated_u >= 0)
                & (cumulated_u < width)
                & (cumulated_v >= 0)
                & (cumulated_v < height)
            )

            cumulated_u_for_depthimage = cumulated_u[mask_for_depthimage]
            cumulated_v_for_depthimage = cumulated_v[mask_for_depthimage]
            cumulated_z_for_depthimage = cumulated_z[mask_for_depthimage]

            cumulated_u_for_pointcloud = cumulated_u[mask_for_pointcloud]
            cumulated_v_for_pointcloud = cumulated_v[mask_for_pointcloud]

            cumulated_p_c = cumulated_p_c[mask_for_pointcloud]
            cumulated_p_c_color = undistorted_image[
                cumulated_v_for_pointcloud, cumulated_u_for_pointcloud
            ].astype("f4") / 255.0
            cumulated_p_c_color = cumulated_p_c_color[:, [2, 1, 0]]

            cumulated_sensor3d_depth[cumulated_sensor3d_depth == 0] = np.inf
            np.minimum.at(
                cumulated_sensor3d_depth,
                (cumulated_v_for_depthimage, cumulated_u_for_depthimage),
                cumulated_z_for_depthimage,
            )
            cumulated_sensor3d_depth[np.isinf(cumulated_sensor3d_depth)] = 0
            mask_for_depthimage = cumulated_sensor3d_depth > 0
            cumulated_v_for_depthimage, cumulated_u_for_depthimage = np.where(mask_for_depthimage)
            cumulated_z_for_depthimage = cumulated_sensor3d_depth[mask_for_depthimage]

            if self.configs["system"]["isFilterDepthByKNN"] is True:
                (
                    cumulated_u_for_depthimage,
                    cumulated_v_for_depthimage,
                    cumulated_z_for_depthimage,
                ) = self.filter_depth_by_knn(
                    cumulated_u_for_depthimage,
                    cumulated_v_for_depthimage,
                    cumulated_z_for_depthimage,
                    k=self.configs["system"]["K_for_filterdepth"],
                )

            cumulated_sensor3d_depth = np.zeros(
                (undistorted_image.shape[0], undistorted_image.shape[1]), dtype="f4"
            )
            cumulated_sensor3d_depth[cumulated_v_for_depthimage, cumulated_u_for_depthimage] = (
                cumulated_z_for_depthimage
            )

            if self.configs["output"]["isSaveVisualizationDepthImage"] is True:
                cumulated_sensor3d_depth_vis = undistorted_image.copy()
                vis_min_depth = self.configs["output"]["minimum_depth_for_vis_depthimage"]
                vis_max_depth = self.configs["output"]["maximum_depth_for_vis_depthimage"]
                if vis_min_depth < 0 or vis_max_depth < 0:
                    vis_min_depth, vis_max_depth = utils.depth_range_by_ratio(
                        cumulated_z_for_depthimage, keep=0.98
                    )
                depth_colors = utils.single_depths2colors(
                    cumulated_z_for_depthimage,
                    vis_min_depth,
                    vis_max_depth,
                )
                cumulated_sensor3d_depth_vis[cumulated_v_for_depthimage, cumulated_u_for_depthimage] = depth_colors

            synchronized_image_data["undistorted_image"] = undistorted_image
            synchronized_image_data["cumulated_p_c"] = cumulated_p_c
            synchronized_image_data["cumulated_p_c_color"] = cumulated_p_c_color
            synchronized_image_data["cumulated_sensor3d_depth"] = cumulated_sensor3d_depth
            if self.configs["output"]["isSaveVisualizationDepthImage"] is True:
                synchronized_image_data["cumulated_sensor3d_depth_vis"] = cumulated_sensor3d_depth_vis

        if self.configs["output"]["isOutput"] is True:
            self.writeSample(sample)

        return sample

    def writeSample(self, sample):
        output_dir = self.configs["output"]["path"]

        for camera_data_idx, synchronized_image_data in enumerate(sample["synchronized_image_data_list"]):
            name_wo_suffix, suffix = os.path.splitext(synchronized_image_data["name"])
            camera_config = self.camera_data_lists[camera_data_idx].config
            output_relative_path_dict = camera_config["output_relative_path_dict"]

            path = os.path.join(output_dir, output_relative_path_dict["undistorted_images_path"], synchronized_image_data["name"])
            if not os.path.exists(path):
                cv2.imwrite(path, synchronized_image_data["undistorted_image"])

            path = os.path.join(output_dir, output_relative_path_dict["GT_depth_output_relative_path"], name_wo_suffix + ".tiff")
            if not os.path.exists(path):
                cv2.imwrite(path, synchronized_image_data["cumulated_sensor3d_depth"])

            path = os.path.join(output_dir, output_relative_path_dict["GT_depth_vis_output_relative_path"], name_wo_suffix + ".png")
            if self.configs["output"]["isSaveVisualizationDepthImage"] is True and not os.path.exists(path):
                cv2.imwrite(path, synchronized_image_data["cumulated_sensor3d_depth_vis"])

            path = os.path.join(output_dir, output_relative_path_dict["GT_pointcloud_output_relative_path"], name_wo_suffix + ".pcd")
            if not os.path.exists(path):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(synchronized_image_data["cumulated_p_c"])
                pcd.colors = o3d.utility.Vector3dVector(synchronized_image_data["cumulated_p_c_color"])
                o3d.io.write_point_cloud(path, pcd, write_ascii=False)

            with open(os.path.join(output_dir, output_relative_path_dict["GT_pose_output_relative_path"], "Twc.txt"), "a") as f:
                ts = synchronized_image_data["ts"]
                t, q = utils.T_to_pose(synchronized_image_data["T_w_cam_idx"])
                if isinstance(ts, str):
                    f.write(
                        f"{ts} {t[0]:.15g} {t[1]:.15g} {t[2]:.15g} "
                        f"{q[0]:.15g} {q[1]:.15g} {q[2]:.15g} {q[3]:.15g}\n"
                    )
                else:
                    f.write(
                        f"{ts:.9f} {t[0]:.15g} {t[1]:.15g} {t[2]:.15g} "
                        f"{q[0]:.15g} {q[1]:.15g} {q[2]:.15g} {q[3]:.15g}\n"
                    )
