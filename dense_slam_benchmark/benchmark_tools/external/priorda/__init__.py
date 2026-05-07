import torch
from prior_depth_anything import PriorDepthAnything
import time




class PriorDepthAnythingWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        version,
        ckpt_dir,
        conditioned_model_size,
        coarse_only,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.model = PriorDepthAnything(
            version=version,
            ckpt_dir=ckpt_dir,
            conditioned_model_size=conditioned_model_size,
            coarse_only=coarse_only,
        )

    def forward(self, frames):


        views = [view for frame in frames for view in frame]


        res = []
        for view in views:

            sparse_mask  = view['input_depth'] > self.model.sampler.min_depth
            cover_mask = torch.zeros_like(sparse_mask)
            
            input_view = {
                'images':view['undistorted_image'],
                'prior_depths': view['input_depth'],
                'sparse_depths': view['input_depth'],
                'sparse_masks': sparse_mask,
                'cover_masks': cover_mask,
                'pattern': None,
                'geometric_depths': None,
            }


            start = time.time()

            output = self.model.forward(
                **input_view
            ) 

            output = output.detach().cpu().squeeze(1).numpy()

            end = time.time()

            runtime = end - start

            # output = output.reshape((height, width))
            res.append({
                'pred_depth': output,
                'pred_depth_mask': output > 0,
                'pred_depth_confidence': (output > 0).astype("float32"),
                'pred_T_w_c': view['T_w_c'].cpu().numpy(),
                'pred_intrinsics': view['intrinsics'].cpu().numpy(),
                'runtime':runtime,
            })

        

        return res
