import torch
from dataclasses import dataclass
from dense_slam_benchmark.dataset_tools import utils
import torchvision
import cv2
import open3d as o3d 
import os
import numpy as np
from PIL import Image
import PIL
from dense_slam_benchmark.benchmark_tools.utils.cropping import (
    crop_resize_if_necessary
)

@dataclass
class ImageNormalization:
    mean: torch.Tensor
    std: torch.Tensor


IMAGE_NORMALIZATION_DICT = {
    "dummy": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "dinov2": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dinov3": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dust3r": ImageNormalization(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
}


class CameraDataset:

    def __init__(self, camera_config):
        super().__init__()

        self.config = camera_config
        self.samples = []

        self.undistorted_image_names = os.listdir(camera_config['datapath']['undistorted_images'])
        sorted(self.undistorted_image_names)
        self.input_depth_names = os.listdir(camera_config['datapath']['input_depth'])
        sorted(self.input_depth_names)
        self.input_pointcloud_names = os.listdir(camera_config['datapath']['input_pointcloud'])
        sorted(self.input_pointcloud_names)
        self.GT_depth_names = os.listdir(camera_config['datapath']['GT_depth'])
        sorted(self.GT_depth_names)
        self.GT_pointcloud_names = os.listdir(camera_config['datapath']['GT_pointcloud'])
        sorted(self.GT_pointcloud_names)

        self.readDatasample()


    def readDatasample(self):

        idx = 0
        with open(self.config['datapath']['input_pose'], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ts, x, y, z, qx, qy, qz, qw = map(float, line.split())

                T_w_c = utils.pose_to_T(x, y, z, qx, qy, qz, qw)
                
                datasample = {
                    "ts": ts, 
                    "T_w_c": T_w_c,
                    "undistorted_image_name": self.undistorted_image_names[idx],
                    "input_depth_name": self.input_depth_names[idx],
                    "input_pointcloud_name": self.GT_pointcloud_names[idx],
                    "GT_depth_name": self.GT_depth_names[idx],
                    "GT_pointcloud_name": self.GT_pointcloud_names[idx],

                }
                self.samples.append(datasample)
                idx+=1
      


    def getGTPointCloud(self, sample_idx):
        
        pcd = o3d.io.read_point_cloud(os.path.join(self.config['datapath']['GT_pointcloud'], self.samples[sample_idx]['GT_pointcloud_name'])) 
        GT_pointcloud = np.asarray(pcd.points, dtype="f4")
        return GT_pointcloud

    def getInputPointCloud(self, sample_idx):
        
        pcd = o3d.io.read_point_cloud(os.path.join(self.config['datapath']['input_pointcloud'], self.samples[sample_idx]['input_pointcloud_name'])) 
        GT_pointcloud = np.asarray(pcd.points, dtype="f4")
        return GT_pointcloud




class Testdataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.is_metric_scale = config['dataset_test']['is_metric_scale']

        self.resolution = config['model']['test_resolution']
        self.camera_dataset_by_id = {}
        self.sample_indexes_per_views_list = []
        self.depth_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
        )


        self.data_norm_type = config['model']['data_norm_type']

        if self.data_norm_type in IMAGE_NORMALIZATION_DICT.keys():
            image_norm = IMAGE_NORMALIZATION_DICT[self.data_norm_type]
            self.image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=image_norm.mean, std=image_norm.std),
                ]
            )

        if self.data_norm_type == 'unchange':

            self.image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.PILToTensor(),
                ]
            )

        for camera_config in config['cameras']:
            camera_dataset = CameraDataset(config['cameras'][camera_config])
            self.camera_dataset_by_id[camera_dataset.config['id']] = camera_dataset


        self.makeSampleIndexPerViewsInSequential()

    

    @staticmethod
    def get_views(views, start_index, require_num_view):
        n = len(views)
        if n == 0:
            return []
        indexes_views = list(range(n))
        return indexes_views[start_index:start_index + require_num_view]

    def makeSampleIndexPerViewsInSequential(self):

        num_view = self.config['num_view_for_sub_scene']
        stride = self.config['stride_for_sub_scene']
        first_camera_dataset = next(iter(self.camera_dataset_by_id.values()))
        Nsamples = len(first_camera_dataset.samples)

        if num_view >= Nsamples:
            self.sample_indexes_per_views_list.append(
                self.get_views(first_camera_dataset.samples, 0, num_view)
            )
            return

        last_start_exclusive = Nsamples - num_view + 1
        last_start = -1
        for i in range(0, last_start_exclusive, stride):
            self.sample_indexes_per_views_list.append(
                self.get_views(first_camera_dataset.samples, i, num_view)
            )
            last_start = i

        tail_start = Nsamples - num_view
        if last_start < tail_start:
            self.sample_indexes_per_views_list.append(
                self.get_views(first_camera_dataset.samples, tail_start, num_view)
            )
             
   

    
    def __len__(self):
        return len(self.sample_indexes_per_views_list)
    
  

    def __getitem__(self, idx):
        
        output_frames = []
        sample_indexes_per_views = self.sample_indexes_per_views_list[idx]
        
        for sample_idx in sample_indexes_per_views:

            frame = []

            for camera_dataset in self.camera_dataset_by_id.values():

                undistorted_image = cv2.imread(os.path.join(camera_dataset.config['datapath']['undistorted_images'], camera_dataset.samples[sample_idx]['undistorted_image_name']))
                input_depth = cv2.imread(os.path.join(camera_dataset.config['datapath']['input_depth'], camera_dataset.samples[sample_idx]['input_depth_name']), cv2.IMREAD_UNCHANGED)
                GT_depth = cv2.imread(os.path.join(camera_dataset.config['datapath']['GT_depth'], camera_dataset.samples[sample_idx]['GT_depth_name']), cv2.IMREAD_UNCHANGED)

                # vis = undistorted_image.copy()

                # # depth valid mask
                # mask = input_depth > 0

                # # draw valid depth pixels in red
                # vis[mask] = (0, 0, 255)   # BGR in OpenCV

                # cv2.imshow("depth points on image", vis)
           

                original_h, original_w = undistorted_image.shape[:2]
                target_w, target_h  = self.resolution



                intrinsics = camera_dataset.config['undistorted_intrinsics']
                K = np.array([
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1]
                ], dtype=np.float32)

                # sx = target_w / original_w
                # sy = target_h / original_h


                # K[0, 0] *= sx  # fx
                # K[1, 1] *= sy  # fy
                # K[0, 2] *= sx  # cx
                # K[1, 2] *= sy  # cy

                # undistorted_raw_image = cv2.resize(undistorted_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                # input_depth = cv2.resize(input_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                # GT_depth = cv2.resize(GT_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


                # vis2 = undistorted_raw_image.copy()

                # # depth valid mask
                # mask = input_depth > 0

                # # draw valid depth pixels in red
                # vis2[mask] = (0, 0, 255)   # BGR in OpenCV

                # cv2.imshow("depth points on image after resize", vis2)

                # cv2.waitKey(0)

                _, input_depth, _ = crop_resize_if_necessary(
                    image=undistorted_image,
                    resolution=self.resolution,
                    depthmap=input_depth,
                    intrinsics=K,
                    additional_quantities=None,
                )

                undistorted_raw_image, GT_depth, K = crop_resize_if_necessary(
                    image=undistorted_image,
                    resolution=self.resolution,
                    depthmap=GT_depth,
                    intrinsics=K,
                    additional_quantities=None,
                )

                undistorted_image = self.image_transform(undistorted_raw_image)
                input_depth = self.depth_transform(input_depth)
                GT_depth = self.depth_transform(GT_depth)
                input_depth_mask = input_depth > 0
                GT_depth_mask = GT_depth > 0    
                
                # input_depth_mask = input_depth > 0
                # GT_depth = cv2.imread(os.path.join(camera_dataset.config['datapath']['GT_depth'], camera_dataset.samples[idx]['GT_depth_name']), cv2.IMREAD_UNCHANGED)
                # pcd = o3d.io.read_point_cloud(os.path.join(camera_dataset.config['datapath']['GT_pointcloud'], camera_dataset.samples[idx]['GT_pointcloud_name'])) 
                # GT_pointcloud = np.asarray(pcd.points, dtype="f4")

                view_data = {
                    "idx": sample_idx,
                    'name': camera_dataset.samples[sample_idx]['undistorted_image_name'],
                    "camera_id": camera_dataset.config['id'],
                    "camera_name": camera_dataset.config['name'],
                    "dataset": self.config['dataset'],
                    "scene_name": self.config['scene_name'],
                    "ts": camera_dataset.samples[sample_idx]["ts"], 
                    "T_w_c": camera_dataset.samples[sample_idx]["T_w_c"],
                    'data_norm_type': self.data_norm_type,
                    'is_metric_scale': self.is_metric_scale,
                    "intrinsics": torch.tensor(K),
                    'undistorted_raw_image': np.array(undistorted_raw_image),
                    "undistorted_image": undistorted_image,
                    "input_depth": input_depth,
                    "input_depth_mask":  input_depth_mask,
                    'GT_depth': GT_depth,
                    'GT_depth_mask': GT_depth_mask,
                    # "input_depth_mask": input_depth_mask,
                    # "GT_depth": GT_depth,
                    # "GT_pointcloud": GT_pointcloud,
                }

                frame.append(view_data)

            output_frames.append(frame)    

        return output_frames
