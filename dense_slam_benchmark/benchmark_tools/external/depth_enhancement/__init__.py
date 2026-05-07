from depth_enhancement_with_sparse_geometry_points.models import PixelWiseCorrection
import torch
import time




class DepthEnhancementWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        depth_model_config,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.model = PixelWiseCorrection(
            depth_model_config,
            **kwargs,
        )

    def forward(self, frames):


        views = [view for frame in frames for view in frame]


        res = []
        for i, view in enumerate(views):

            input_view = {
                'images':view['undistorted_image'],
                'sparse_depths': view['input_depth'],
                'sparse_depth_masks': view['input_depth_mask'],
            }

            # if i < 4:
            #     continue


            start = time.time()

            output = self.model(
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
                'runtime':runtime,
            })

        

        return res

