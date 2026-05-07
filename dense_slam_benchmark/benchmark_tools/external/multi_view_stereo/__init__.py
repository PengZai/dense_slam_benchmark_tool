from . import multi_view_stereo_pybind
import numpy as np
import cv2 
import time


class MVSWrapper():
    def __init__(
        self,
        name,
        half_ws,
        debug_plot,
        start_match_u,
        start_match_v,
        ACCEPTABLE_MINI_COST,
        ACCEPTABLE_COST_DIFF,
        ACCEPTABLE_DEPTH_PARAMETER,
        MAXIMUM_AGGREAGTE_COST_PENALTY,
        MAXIMUM_RECONSTRUCTION_DISTANCE,
        MAXIMUM_EXTENSION,
        MINIMUM_CORRECT_CONSISTENCY_CHECK,
        min_depth,
        max_depth,
        **kwargs
    ):
        super().__init__()

        self.name = name
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.MINIMUM_CORRECT_CONSISTENCY_CHECK = MINIMUM_CORRECT_CONSISTENCY_CHECK

        # Init the model and load the checkpoint
        self.model = multi_view_stereo_pybind.MultiViewStereo(
            half_ws,
            debug_plot,
            start_match_u,
            start_match_v,
            ACCEPTABLE_MINI_COST,
            ACCEPTABLE_COST_DIFF,
            ACCEPTABLE_DEPTH_PARAMETER,
            MAXIMUM_AGGREAGTE_COST_PENALTY,
            MAXIMUM_RECONSTRUCTION_DISTANCE,
            MAXIMUM_EXTENSION,
            MINIMUM_CORRECT_CONSISTENCY_CHECK,
        )

    def __call__(self, frames):

        # convert multi camera in single frame to views
        num_frame = len(frames)

        cameras = {}
        batch_size_per_view, _, height, width = frames[0][0]["undistorted_image"].shape

        num_views_per_frame = len(frames[0])
        for view in frames[0]:
            camera = multi_view_stereo_pybind.Camera()
            camera.setId(view['camera_id'].item())
            camera.setName(view['camera_name'][0])
            camera.setOriginalResolution([width, height])
            camera.setResolution([width, height])
            camera.setCameraModelName("pinhole")
            camera.setOriginalIntrinsicMatrix(view['intrinsics'].squeeze().cpu().numpy())
            camera.setOriginalDistortionCoeffs([0,0,0,0])
            camera.setDistortionModelName("radtan")
            camera.createUndistortModel()
            cameras[view['camera_id'].item()] = camera

        views = [view for frame in frames for view in frame]


        assert batch_size_per_view == 1, (
            f"Batch size of input views should be 1, but got {batch_size_per_view}."
        )

        images = []
        for view in views:

            image = multi_view_stereo_pybind.Image()
            image.setName(view['name'][0])
            image.setCamera(cameras[view['camera_id'].item()])
            image.setHeightOrg(height)
            image.setWidthOrg(width)
            image.setHeight(height)
            image.setWidth(width)
            image.setPoseId(view['idx'].item())
            image.setTransformationMatrix(view['T_w_c'].squeeze().cpu().numpy())
            image.setPresetInitMaxDepth(self.max_depth)
            image.setPresetInitMinDepth(self.min_depth)

            input_image = view["undistorted_image"].squeeze().cpu().numpy()
            input_image = np.transpose(input_image, (1, 2, 0))
            image.loadData(view["undistorted_image"].squeeze().cpu().numpy())
            # bgr_data = image.getNumpyBGRData()
            # cv2.imshow("bgr_data", bgr_data)
            # cv2.imshow("input_image", input_image)

            # cv2.waitKey(0)
            images.append(image)


        
        start = time.time()

        self.model.scene_reconstruction(images)
        self.model.consistency_check(images)

        end = time.time()

        runtime = end - start

        res = []
        for frame_idx in range(num_frame):

            pred_idx = frame_idx*num_views_per_frame
            image = images[pred_idx]
            depth = image.getEigenMatrixDepth(1.0, self.MINIMUM_CORRECT_CONSISTENCY_CHECK)
            depth = np.expand_dims(depth, axis = 0)
            res.append(
                {
                    'pred_depth':depth,
                    'pred_depth_mask': (depth > 0), # this 1 threshold according to scene.show() visualization setting
                    'pred_depth_confidence': (depth > 0).astype("float32"),
                    'pred_T_w_c': view['T_w_c'].cpu().numpy(),
                    'runtime': runtime / float(num_frame)
                }
            )

        return res







# image = multi_view_stereo_pybind.Image()

# depth = image.getEigenMatrixDepth(0.5, 0)
# print(depth.shape)
# print("end")
