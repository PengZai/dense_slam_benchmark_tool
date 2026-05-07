import os
import tempfile
import warnings

import torch

from dust3r.image_pairs import make_pairs

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import load_model
import time



# sparse_ga_optim_level, "refine", "refine+depth",  "refine+depth+intrinsics",
class MASt3RSGWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_path,
        cache_dir,
        scene_graph="complete",
        sparse_ga_lr1=0.07,
        sparse_ga_niter1=300,
        sparse_ga_lr2=0.01,
        sparse_ga_niter2=300,
        sparse_ga_optim_level="refine+depth",
        sparse_ga_shared_intrinsics=False,
        sparse_ga_matching_conf_thr=5.0,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.ckpt_path = ckpt_path
        self.cache_dir = cache_dir
        self.scene_graph = scene_graph
        self.sparse_ga_lr1 = sparse_ga_lr1
        self.sparse_ga_niter1 = sparse_ga_niter1
        self.sparse_ga_lr2 = sparse_ga_lr2
        self.sparse_ga_niter2 = sparse_ga_niter2
        self.sparse_ga_optim_level = sparse_ga_optim_level
        self.sparse_ga_shared_intrinsics = sparse_ga_shared_intrinsics
        self.sparse_ga_matching_conf_thr = sparse_ga_matching_conf_thr
    

        # Init the model and load the checkpoint
        self.model = load_model(self.ckpt_path, device="cpu")

    def forward(self, frames):

        # convert multi camera in single frame to views
        num_frame = len(frames)

        num_views_per_frame = len(frames[0])
        views = [view for frame in frames for view in frame]

        batch_size_per_view, _, height, width = views[0]["undistorted_image"].shape

        device = views[0]["undistorted_image"].device
        assert batch_size_per_view == 1, (
            f"Batch size of input views should be 1, but got {batch_size_per_view}."
        )


        images = []
        image_paths = []
        init={}
        for view in views:
            images.append(
                dict(
                    img=view["undistorted_image"].cpu(),
                    idx=len(images),
                    instance=str(len(images)),
                    true_shape=torch.tensor(view["undistorted_image"].shape[-2:])[None]
                    .numpy(),
                )
            )
            image_path = os.path.join(view['scene_name'][0], view['camera_name'][0], 'undistorted_images', view['name'][0])
            image_paths.append(image_path)
            init[image_path] = {}
            if 'intrinsics' in view:
                init[image_path]['intrinsics'] = view['intrinsics'].squeeze()

            # it looks like it don't support init sparse depth and init pose, so don't bother to try
            #  depth = init_values.get('depthmap')
            #  cam2w = init_values.get('cam2w')


        pairs = make_pairs(
            images, scene_graph=self.scene_graph, prefilter=None, symmetrize=True
        )

        start = time.time()

        with torch.enable_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                tempfile.mkdtemp(dir=self.cache_dir)
                scene = sparse_global_alignment(
                    image_paths,
                    pairs,
                    self.cache_dir,
                    self.model,
                    lr1=self.sparse_ga_lr1,
                    niter1=self.sparse_ga_niter1,
                    lr2=self.sparse_ga_lr2,
                    niter2=self.sparse_ga_niter2,
                    device=device,
                    opt_pp="intrinsics" in self.sparse_ga_optim_level,
                    opt_depth="depth" in self.sparse_ga_optim_level,
                    shared_intrinsics=self.sparse_ga_shared_intrinsics,
                    matching_conf_thr=self.sparse_ga_matching_conf_thr,
                    verbose=False,
                    init = init,
                )

        # Make sure scene is not None
        if scene is None:
            raise RuntimeError("Global optimization failed.")

        intrinsics = scene.intrinsics
        c2w_poses = scene.get_im_poses()
        pts3d, depths, confs = scene.get_dense_pts3d()
        
        end = time.time()

        runtime = end - start

        res = []
        for frame_idx in range(num_frame):

            pred_idx = frame_idx*num_views_per_frame
            pred_depth = depths[pred_idx].reshape((height, width)).unsqueeze(0).detach().cpu().numpy()
            pred_depth_mask = (confs[pred_idx] > 1).unsqueeze(0).detach().cpu().numpy()
            pred_depth_confidence = confs[pred_idx].reshape((height, width)).unsqueeze(0).detach().cpu().numpy()
            pred_T_w_c = c2w_poses[pred_idx].unsqueeze(0).detach().cpu().numpy()
            pred_intrinsics = intrinsics[pred_idx].unsqueeze(0).detach().cpu().numpy()
            res.append(
                {
                    'pred_depth': pred_depth,
                    'pred_depth_mask': pred_depth_mask, # this 1 threshold according to scene.show() visualization setting
                    'pred_depth_confidence': pred_depth_confidence,
                    'pred_T_w_c': pred_T_w_c,
                    'pred_intrinsics': pred_intrinsics,
                    'runtime': runtime / float(num_frame) # unit second
                }
            )

        # scene.show()

        return res


