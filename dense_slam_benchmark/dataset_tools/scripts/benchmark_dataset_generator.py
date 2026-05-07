import weakref
import moderngl_window
from dense_slam_benchmark.dataset_tools.visualizer import Visualizer
import argparse
from moderngl_window.timers.clock import Timer
import yaml
from dense_slam_benchmark.dataset_tools.datasets import build_dataset
import open3d as o3d
import os 
import numpy as np
from dense_slam_benchmark.dataset_tools import utils 



    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--configdir", default="configs/dataset_tools/BotanicGarden.yaml", type=str, help="path to configure directory")
    args = parser.parse_args()


    VisualizerConfig = Visualizer

    with open(args.configdir, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

  
    data_source_idx = configs['system']['use_data_source']
    data_source = configs['data_source'+str(data_source_idx)]

    dataset = build_dataset(configs, config_path=args.configdir)


    invalid_data_count = 0
    for sample_idx, sample in enumerate(dataset.samples):
        if not dataset.is_sample_idx_selected(sample_idx):
            continue
        print("processing sample ", sample_idx)

        sample = dataset.loadBenchmarkData(sample)
        if sample == None:
           invalid_data_count+=1
    
    dataset.samples = dataset.get_selected_samples()
    print("invalid_data_count:",invalid_data_count)   
    


    # if "sensor3dtype" in data_source:
    #     if data_source["sensor3dtype"] == 'lidarpointcloud':
    #         poincloud_processor = LidarPointCloud(configs, dataset)
    #     elif data_source["sensor3dtype"] == 'colorizedpointcloud':
    #         poincloud_processor = ColoarizedPointCloud(configs, dataset)
    #     elif data_source["sensor3dtype"] == 'imagedepth':
    #         poincloud_processor = ImageDepthPointCloud(configs, dataset)

    #     points_w, colors, poses = poincloud_processor.run()

    reference_frame_points_list = []
    colors_list = []
    reference_frame_pose_list = []
    pose_lines_by_camera_id = {
        camera_data.id: [] for camera_data in dataset.camera_data_lists
    }
    reference_camera_entry_idx = data_source['used_camera_idxes'][0]
    reference_camera_config = configs['cameras']['camera' + str(reference_camera_entry_idx)]
    reference_camera_name = reference_camera_config['name']
    T_w_reference_camera_at_first_sample = dataset.samples[0]['synchronized_image_data_list'][0]['T_w_cam_idx']
    # T_w_reference_camera_at_first_sample = np.eye(4)

    for sample_idx, sample in enumerate(dataset.samples):
        for synchronized_image_data in sample['synchronized_image_data_list']:
            T_w_cam_idx = synchronized_image_data['T_w_cam_idx']
            synchronized_p_c = synchronized_image_data['cumulated_p_c']
            points_c = synchronized_p_c
            points_c_h = np.hstack([points_c, np.ones((points_c.shape[0], 1), dtype=np.float32)])  # (N,4)
            T_reference_camera_at_first_sample_cam_idx = (
                np.linalg.inv(T_w_reference_camera_at_first_sample) @ T_w_cam_idx
            )
            points_in_reference_frame_h = (T_reference_camera_at_first_sample_cam_idx @ points_c_h.T).T
            colors = synchronized_image_data['cumulated_p_c_color']
            reference_frame_points_list.append(points_in_reference_frame_h[:, :3])
            colors_list.append(colors)
            reference_frame_pose_list.append(T_reference_camera_at_first_sample_cam_idx)
            t, q = utils.T_to_pose(T_reference_camera_at_first_sample_cam_idx)
            ts = synchronized_image_data['ts']
            if isinstance(ts, str):
                ts_str = ts
            else:
                ts_str = f"{ts:.9f}"
            pose_lines_by_camera_id[synchronized_image_data['camera_id']].append(
                f"{ts_str} {t[0]:.15g} {t[1]:.15g} {t[2]:.15g} "
                f"{q[0]:.15g} {q[1]:.15g} {q[2]:.15g} {q[3]:.15g}\n"
            )

        # pose_list.append(sample['T_w_p'])


    reference_frame_points = np.vstack(reference_frame_points_list, dtype='f4')
    reference_frame_points = np.ascontiguousarray(reference_frame_points)

    colors = np.vstack(colors_list, dtype='f4')
    colors = np.ascontiguousarray(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reference_frame_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(
        os.path.join(configs['output']['path'], reference_camera_config['output_relative_path_dict']['name_for_GT_dir']) + '.pcd',
        pcd,
        write_ascii=False,
    )

    for camera_data in dataset.camera_data_lists:
        pose_output_path = os.path.join(
            configs['output']['path'],
            camera_data.config['name'],
            f"aligned_{data_source['trajectoryname']}_pose.txt",
        )
        with open(pose_output_path, "w", encoding="utf-8") as f:
            f.write("#timestamp/index x y z q_x q_y q_z q_w\n")
            f.writelines(pose_lines_by_camera_id[camera_data.id])


    poses = np.stack(reference_frame_pose_list, dtype="f4")
    rgbas = np.hstack([colors, np.ones((colors.shape[0], 1), dtype=np.float32)])  # (N,4)



    if configs['visualization']['isVisualization'] == True:
        VisualizerConfig._points = reference_frame_points
        VisualizerConfig._rgbas = rgbas
        VisualizerConfig._poses = poses

        window_cls = moderngl_window.get_local_window_cls("glfw")
        window = window_cls(
            title=VisualizerConfig._title,
            size=VisualizerConfig._window_size,
            fullscreen=False,
            resizable=True,
            visible=True,
            gl_version=(3, 3),
            aspect_ratio=None,
            vsync=True,
            samples=4,
            cursor=True,
            backend="glfw",
        )
        window.print_context_info()
        moderngl_window.activate_context(window=window)
        window.ctx.gc_mode = "auto"
        timer = Timer()

        window_config = VisualizerConfig(
            ctx=window.ctx,
            wnd=window,
            timer=timer,
        )
        window._config = weakref.ref(window_config)


        window.swap_buffers()
        window.set_default_viewport()

        timer.start()

        while not window.is_closing:
            current_time, delta = timer.next_frame()

            if window_config.clear_color is not None:
                window.clear(*window_config.clear_color)

            # Always bind the window framebuffer before calling render
            window.use()


            window.render(current_time, delta)
            window.swap_buffers()

    print("end")
