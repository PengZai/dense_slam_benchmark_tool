import torch
import numpy as np
import time

SIMILARITY_REFINE_MIN_CORRESPONDENCES = 10

def update_depth_confidence_from_mask(result_dict):
    pred = result_dict['pred']
    mask = pred['depth_mask']
    confidence = pred.get('depth_confidence')
    if confidence is None or confidence.shape != mask.shape:
        confidence = mask.astype(np.float32)
    else:
        confidence = confidence.astype(np.float32, copy=True)
        confidence[~mask] = 0.0
    pred['depth_confidence'] = confidence




def simple_postprocess(config, result_list):
    similarity_refine_mode = getattr(
        config.model.postprocessing,
        "similarity_refine_mode",
        "sub_scene",
    )
    if similarity_refine_mode not in {"depth", "sub_scene"}:
        raise ValueError(
            "Unknown postprocessing.similarity_refine_mode: "
            f"{similarity_refine_mode}. Expected 'depth' or 'sub_scene'."
        )

    sub_scene_cfg = getattr(config.model.postprocessing, "sub_scene", None)

    if similarity_refine_mode == "sub_scene":
        if sub_scene_cfg is None or getattr(sub_scene_cfg, "isAlignSubScenesSequential", True):
            align_sub_scenes_sequential(result_list)

    # Flatten nested sub-scenes and dedupe by sample_idx so each frame appears once.
    deduped_by_sample_idx = {}
    for result_list_per_sub_scene in result_list:
        for result_dict in result_list_per_sub_scene:
            deduped_by_sample_idx[result_dict['basic']['sample_idx']] = result_dict
    result_list = list(deduped_by_sample_idx.values())



    start = time.time()
    for result_dict in result_list:
        update_pred_pts3d(result_dict)

    # "sub_scene" aligns the whole scene with a similarity transform.
    # "depth" refines each depth map independently with affine depth fitting.
    if config.model.postprocessing.isAlignWithGT == True:
        if similarity_refine_mode == "sub_scene":
            gt_align_method = (
                getattr(sub_scene_cfg, "gt_align_method", "pointcloud_icp")
                if sub_scene_cfg is not None
                else "pointcloud_icp"
            )
            if gt_align_method == "depth_pixel_sim3":
                refine_similarity_via_gt_depth(result_list)
            elif gt_align_method == "pointcloud_icp":
                similarity_refine_use_pose_init = (
                    getattr(sub_scene_cfg, "similarity_refine_use_pose_init", True)
                    if sub_scene_cfg is not None
                    else True
                )
                refine_similarity_sub_scene(
                    result_list,
                    similarity_refine_use_pose_init,
                )
            else:
                raise ValueError(
                    "Unknown postprocessing.sub_scene.gt_align_method: "
                    f"{gt_align_method}. Expected 'pointcloud_icp' or "
                    "'depth_pixel_sim3'."
                )
        else:
            refine_affine_depth_results(result_list)

    for result_dict in result_list:
        if config.model.postprocessing.isJustCompareNearDistance == True:

            result_dict['pred']['depth_mask'] = result_dict['pred']['depth_mask'] & ((result_dict['pred']['depth'] <= config.model.postprocessing.maximum_near_distance) & ((result_dict['pred']['depth'] > 0))).squeeze()

        else:
            result_dict['pred']['depth_mask'] = result_dict['pred']['depth_mask'] & (result_dict['pred']['depth'] > 0).squeeze()

        # print(f"Postprocessing : {result_dict['pred']['depth_mask'].sum()} valid pixels after postprocessing.")

    for result_dict in result_list:
        update_pred_pts3d(result_dict)

    postprocess_time = time.time() - start
    start = time.time()
    if config.model.postprocessing.isConsistencyCheck == True:

        consistency_check(result_list)

    postprocess_time += time.time() - start

    for result_dict in result_list:
        update_depth_confidence_from_mask(result_dict)

        result_dict['pred']['postprocess_time'] = postprocess_time/float(len(result_list))


    return result_list

def make_pts3d(depth, K_matrix, mask):
    # If depth is HxWx1, squeeze it to HxW
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth_2d = depth[..., 0]
    else:
        depth_2d = depth

    h, w = depth_2d.shape

    fx = K_matrix[0, 0]
    fy = K_matrix[1, 1]
    cx = K_matrix[0, 2]
    cy = K_matrix[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    x = (u - cx) * depth_2d / fx
    y = (v - cy) * depth_2d / fy
    z = depth_2d

    pts3d = np.stack((x, y, z), axis=-1).astype(depth.dtype)

    return pts3d


def update_pred_pts3d(result_dict):
    result_dict['pred']['pts3d'] = make_pts3d(
        result_dict['pred']['depth'],
        result_dict['pred']['intrinsics'],
        result_dict['pred']['depth_mask'],
    )


def AffineRefinefitting(gt, pred, mask):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.enable_grad():
        # example data
        x = torch.tensor(pred).flatten().to(device)
        y = torch.tensor(gt).flatten().to(device)
        mask = torch.tensor(mask).flatten().to(device)

        y_valid = y[mask]

        # parameters to learn
        raw_a = torch.tensor(1.0, device=device, requires_grad=True)
        raw_b = torch.tensor(0.0, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([raw_a, raw_b], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,   # decay every 500 iterations
            gamma=0.9        # multiply lr by 0.5
        )

        
        loss_fn = torch.nn.HuberLoss(delta=1.0, reduction="none")

        for i in range(200):
            optimizer.zero_grad()
            y_pred = raw_a * x[mask] + raw_b
            # y_pred = raw_a * x[mask]

            reg_loss = 20.0 / (raw_a + 1e-3).mean()
            # reg_loss = 0

            # Larger weight when gt is close to zero
            # weights = 10 / (torch.abs(y_valid) + 1e-3)
            min_y_valid = y_valid.min()
            max_y_valid = y_valid.max()
            norm_y_valid = (y_valid - min_y_valid) / (max_y_valid - min_y_valid)
            norm_y_valid = torch.clamp(norm_y_valid, 0.0, 1.0)
            weights = 1.0 * (1 - norm_y_valid)

            data_loss = (weights * loss_fn(y_pred, y_valid)).mean()

            loss = data_loss + reg_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"{i}, loss:{loss.item()}, data_loss:{data_loss.item()}, reg_loss:{reg_loss.item()}, a:{raw_a.item()}, b:{raw_b.item()}, lr:{current_lr}")           

        refine_pred = raw_a*x + raw_b
        refine_pred = refine_pred.detach().cpu().numpy()

        near_mask = y < 10
        mae = torch.mean(torch.abs(raw_a * x[mask & near_mask] + raw_b - y[mask & near_mask]))

        print(f"mask: {mask.sum()}, a:{raw_a}, b:{raw_b}, loss:{loss.item()}, mae:{mae}")
        return refine_pred.reshape(pred.shape)

def estimate_open3d_icp_similarity_transform(source_points, target_points, min_correspondences=SIMILARITY_REFINE_MIN_CORRESPONDENCES, init_transform=None):
    if source_points.shape[0] < min_correspondences or target_points.shape[0] < min_correspondences:
        return None

    import open3d as o3d

    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if init_transform is None:
        init_transform = estimate_centroid_scale_transform(source_points, target_points)
    if init_transform is None:
        return None

    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    target_extent = np.linalg.norm(target_points.max(axis=0) - target_points.min(axis=0))
    icp_distance_threshold = max(target_extent, 1.0)

    registration = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        float(icp_distance_threshold),
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    if len(registration.correspondence_set) < min_correspondences:
        return None

    scale, R, t = decompose_sim3_transform(registration.transformation)
    if scale is None:
        return None
    return registration.transformation, scale, R, t


def decompose_sim3_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    scale = float(np.mean(np.linalg.norm(transform[:3, :3], axis=0)))
    if scale <= 1e-12:
        return None, None, None
    R = transform[:3, :3] / scale
    t = transform[:3, 3]
    return scale, R, t


def estimate_centroid_scale_transform(source_points, target_points):
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_radius = np.sqrt(((source_points - source_center) ** 2).sum(axis=1).mean())
    target_radius = np.sqrt(((target_points - target_center) ** 2).sum(axis=1).mean())
    if source_radius <= 1e-8:
        return None

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= target_radius / source_radius
    transform[:3, 3] = target_center - transform[:3, :3] @ source_center
    return transform


def estimate_pose_centers_similarity_transform(source_centers, target_centers):
    source_centers = np.asarray(source_centers[:, :3], dtype=np.float64)
    target_centers = np.asarray(target_centers[:, :3], dtype=np.float64)
    if source_centers.shape != target_centers.shape or source_centers.shape[0] < 3:
        return None

    source_centroid = np.mean(source_centers, axis=0)
    target_centroid = np.mean(target_centers, axis=0)
    source_centered = source_centers - source_centroid
    target_centered = target_centers - target_centroid

    covariance = source_centered.T @ target_centered / source_centers.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)
    V = Vt.T
    sign = np.sign(np.linalg.det(V @ U.T))
    if sign == 0:
        sign = 1.0

    D = np.diag([1.0, 1.0, sign])
    R = V @ D @ U.T
    source_var = np.var(source_centers, axis=0).sum()
    if source_var <= 1e-9:
        return None

    scale = np.sum(singular_values * np.diag(D)) / source_var
    t = target_centroid - scale * R @ source_centroid

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t
    return transform, scale, R, t


def estimate_pose_alignment_transform(result_list_per_sub_scene):
    source_poses = []
    target_poses = []
    source_centers = []
    target_centers = []
    for result_dict in result_list_per_sub_scene:
        source_T_w_c = np.asarray(result_dict['pred']['T_w_c'], dtype=np.float64)
        target_T_w_c = np.asarray(result_dict['GT']['T_w_c'], dtype=np.float64)
        if not (
            np.isfinite(source_T_w_c).all()
            and np.isfinite(target_T_w_c).all()
        ):
            continue
        source_poses.append(source_T_w_c)
        target_poses.append(target_T_w_c)
        source_centers.append(source_T_w_c[:3, 3])
        target_centers.append(target_T_w_c[:3, 3])

    if len(source_centers) >= 3:
        return estimate_pose_centers_similarity_transform(
            np.asarray(source_centers, dtype=np.float64),
            np.asarray(target_centers, dtype=np.float64),
        )
    if len(source_centers) >= 2:
        transform = estimate_centroid_scale_transform(
            np.asarray(source_centers, dtype=np.float64),
            np.asarray(target_centers, dtype=np.float64),
        )
        if transform is None:
            return None
        scale, R, t = decompose_sim3_transform(transform)
        return transform, scale, R, t
    if len(source_centers) > 0:
        transform = target_poses[0] @ np.linalg.inv(source_poses[0])
        scale, R, t = decompose_sim3_transform(transform)
        return transform, scale, R, t
    return None


def estimate_pointcloud_alignment_transform(source_points, target_points, init_transform=None):
    return estimate_open3d_icp_similarity_transform(
        source_points,
        target_points,
        init_transform=init_transform,
    )


def refine_affine_depth_single_result(result_dict):
    pred_mask = result_dict['pred']['depth_mask']
    target_mask = result_dict['GT']['GT_depth_mask']
    affine_mask = pred_mask & target_mask
    if affine_mask.sum() < SIMILARITY_REFINE_MIN_CORRESPONDENCES:
        return

    result_dict['pred']['depth'] = AffineRefinefitting(
        result_dict['GT']['GT_depth'],
        result_dict['pred']['depth'],
        affine_mask,
    ).astype(result_dict['pred']['depth'].dtype)
    result_dict['pred']['affine_depth_refine'] = True


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def pred_points_from_result(result_dict):
    pred_mask = result_dict['pred']['depth_mask']
    if not pred_mask.any():
        return np.empty((0, 3), dtype=np.float32), pred_mask
    return result_dict['pred']['pts3d'][pred_mask], pred_mask


def gt_pointcloud_from_result(result_dict):
    if 'GT_pointcloud' not in result_dict['GT']:
        raise KeyError(
            "GT_pointcloud is required for isAlignWithGT."
        )
    gt_pointcloud = np.asarray(result_dict['GT']['GT_pointcloud'], dtype=np.float32)
    valid = np.isfinite(gt_pointcloud).all(axis=1)
    return gt_pointcloud[valid]


def refine_affine_depth_results(result_list_per_sub_scene):
    for result_dict in result_list_per_sub_scene:
        refine_affine_depth_single_result(result_dict)


def weighted_align_point_maps(points_src, conf_src, points_tgt, conf_tgt,
                              mask=None, conf_threshold=0.0,
                              huber_delta=0.1, max_iters=5, tol=1e-9):
    """Robust confidence-weighted Sim(3) alignment via IRLS with Huber
    re-weighting. Returns (s, R, t) such that points_tgt ≈ s * R @ points_src + t.

    Mirrors VGGT-Long's robust_weighted_estimate_sim3: initialize with
    confidence-weighted Umeyama, then refit each iteration with
    weights = confidence * Huber(residual, huber_delta), stopping when the
    parameters or the weighted Huber loss stop changing.
    """
    src = np.asarray(points_src, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(points_tgt, dtype=np.float64).reshape(-1, 3)
    cs = np.asarray(conf_src, dtype=np.float64).reshape(-1)
    ct = np.asarray(conf_tgt, dtype=np.float64).reshape(-1)

    valid = (
        np.isfinite(src).all(axis=1)
        & np.isfinite(tgt).all(axis=1)
        & (cs > conf_threshold)
        & (ct > conf_threshold)
    )
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)

    if int(valid.sum()) < 3:
        return 1.0, np.eye(3), np.zeros(3)

    src = src[valid]
    tgt = tgt[valid]
    init_weights = np.sqrt(np.maximum(cs[valid] * ct[valid], 0.0))
    if init_weights.sum() <= 1e-12:
        init_weights = np.ones(src.shape[0], dtype=np.float64)

    s, R, t = _weighted_umeyama_sim3(src, tgt, init_weights)

    prev_error = float('inf')
    rot_tol = np.radians(0.1)
    for _ in range(max_iters):
        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)

        huber_w = np.ones_like(residuals)
        large = residuals > huber_delta
        huber_w[large] = huber_delta / np.maximum(residuals[large], 1e-12)

        combined = init_weights * huber_w
        if combined.sum() <= 1e-12:
            break

        s_new, R_new, t_new = _weighted_umeyama_sim3(src, tgt, combined)

        param_change = abs(s_new - s) + float(np.linalg.norm(t_new - t))
        rot_trace = (np.trace(R_new @ R.T) - 1.0) / 2.0
        rot_angle = float(np.arccos(min(1.0, max(-1.0, rot_trace))))

        huber_loss = np.where(
            residuals <= huber_delta,
            0.5 * residuals ** 2,
            huber_delta * (residuals - 0.5 * huber_delta),
        )
        current_error = float((huber_loss * init_weights).sum())

        if (param_change < tol and rot_angle < rot_tol) or (
            abs(prev_error - current_error) < tol * max(prev_error, 1e-12)
        ):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error

    return s, R, t


def _weighted_umeyama_sim3(src, tgt, weights):
    w = weights / weights.sum()
    src_centroid = (w[:, None] * src).sum(axis=0)
    tgt_centroid = (w[:, None] * tgt).sum(axis=0)
    src_d = src - src_centroid
    tgt_d = tgt - tgt_centroid

    H = (w[:, None] * tgt_d).T @ src_d
    U, S_diag, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt

    src_var = (w * (src_d ** 2).sum(axis=1)).sum()
    if src_var <= 1e-12:
        return 1.0, R, tgt_centroid - src_centroid
    s = float((S_diag * np.diag(D)).sum() / src_var)
    t = tgt_centroid - s * R @ src_centroid
    return s, R, t


def align_sub_scenes_sequential(result_list):
    """Align each sub-scene to sub-scene-0's frame using overlapping frames
    (matched by sample_idx). Mirrors VGGT-Long's chunk-and-align: pairwise
    Sim(3) between adjacent sub-scenes, chained so every sub-scene ends up
    in sub-scene-0 coordinates.
    """
    if len(result_list) < 2:
        return result_list

    for sub_scene in result_list:
        for rd in sub_scene:
            update_pred_pts3d(rd)

    pairwise = []
    for k in range(len(result_list) - 1):
        prev_by_idx = {rd['basic']['sample_idx']: rd for rd in result_list[k]}

        src_pts_chunks, tgt_pts_chunks = [], []
        src_conf_chunks, tgt_conf_chunks = [], []

        for curr_rd in result_list[k + 1]:
            sample_idx = curr_rd['basic']['sample_idx']
            prev_rd = prev_by_idx.get(sample_idx)
            if prev_rd is None:
                continue

            curr_T = np.asarray(curr_rd['pred']['T_w_c'], dtype=np.float64)
            prev_T = np.asarray(prev_rd['pred']['T_w_c'], dtype=np.float64)
            curr_world = transform_points(
                curr_rd['pred']['pts3d'].reshape(-1, 3), curr_T,
            )
            prev_world = transform_points(
                prev_rd['pred']['pts3d'].reshape(-1, 3), prev_T,
            )

            curr_mask = np.asarray(curr_rd['pred']['depth_mask']).reshape(-1)
            prev_mask = np.asarray(prev_rd['pred']['depth_mask']).reshape(-1)
            both_valid = curr_mask & prev_mask
            if not both_valid.any():
                continue

            curr_conf = curr_rd['pred'].get('depth_confidence')
            if curr_conf is None:
                curr_conf = curr_mask.astype(np.float32)
            prev_conf = prev_rd['pred'].get('depth_confidence')
            if prev_conf is None:
                prev_conf = prev_mask.astype(np.float32)

            src_pts_chunks.append(curr_world[both_valid])
            tgt_pts_chunks.append(prev_world[both_valid])
            src_conf_chunks.append(
                np.asarray(curr_conf, dtype=np.float64).reshape(-1)[both_valid]
            )
            tgt_conf_chunks.append(
                np.asarray(prev_conf, dtype=np.float64).reshape(-1)[both_valid]
            )

        if len(src_pts_chunks) == 0:
            pairwise.append((1.0, np.eye(3), np.zeros(3)))
            continue

        src_pts = np.concatenate(src_pts_chunks, axis=0)
        tgt_pts = np.concatenate(tgt_pts_chunks, axis=0)
        src_conf = np.concatenate(src_conf_chunks, axis=0)
        tgt_conf = np.concatenate(tgt_conf_chunks, axis=0)

        median_conf = min(float(np.median(src_conf)), float(np.median(tgt_conf)))
        conf_threshold = 0.1 * median_conf if median_conf > 0 else 0.0

        s, R, t = weighted_align_point_maps(
            src_pts, src_conf, tgt_pts, tgt_conf,
            conf_threshold=conf_threshold,
        )
        pairwise.append((s, R, t))

    accumulated = [(1.0, np.eye(3), np.zeros(3))]
    for s_p, R_p, t_p in pairwise:
        s_a, R_a, t_a = accumulated[-1]
        new_s = s_a * s_p
        new_R = R_a @ R_p
        new_t = s_a * (R_a @ t_p) + t_a
        accumulated.append((new_s, new_R, new_t))

    for k, sub_scene in enumerate(result_list):
        s, R, t = accumulated[k]
        for rd in sub_scene:
            T = np.asarray(rd['pred']['T_w_c'], dtype=np.float64)
            T_new = np.eye(4, dtype=np.float64)
            T_new[:3, :3] = R @ T[:3, :3]
            T_new[:3, 3] = s * (R @ T[:3, 3]) + t
            rd['pred']['T_w_c_before_sub_scene_align'] = rd['pred']['T_w_c']
            rd['pred']['T_w_c'] = T_new
            rd['pred']['similarity_scale'] = s
            rd['pred']['similarity_R'] = R
            rd['pred']['similarity_t'] = t


def refine_similarity_via_gt_depth(result_list_per_sub_scene):
    """Align the predicted scene to the GT scene via confidence-weighted
    Sim(3) Umeyama on per-pixel pred ↔ GT correspondences.

    Mirrors align_sub_scenes_sequential, but the target is the GT 3D point
    cloud built from GT_depth + GT intrinsics + GT_T_w_c, instead of an
    adjacent sub-scene. Pred and GT share sample_idx and pixel layout, so
    correspondences are direct (no nearest-neighbour search / ICP).
    """
    if len(result_list_per_sub_scene) == 0:
        return

    src_pts_chunks, tgt_pts_chunks = [], []
    src_conf_chunks, tgt_conf_chunks = [], []

    for rd in result_list_per_sub_scene:
        gt_block = rd.get('GT', {})
        gt_depth = gt_block.get('GT_depth')
        if gt_depth is None:
            continue
        gt_intr = np.asarray(gt_block['intrinsics'], dtype=np.float64)
        gt_mask_2d = gt_block.get('GT_depth_mask')
        if gt_mask_2d is None:
            gt_depth_2d = (
                gt_depth[..., 0]
                if (gt_depth.ndim == 3 and gt_depth.shape[-1] == 1)
                else gt_depth
            )
            gt_mask_2d = gt_depth_2d > 0
        gt_mask_2d = np.asarray(gt_mask_2d, dtype=bool)

        gt_pts3d = make_pts3d(gt_depth, gt_intr, gt_mask_2d)
        T_w_c_gt = np.asarray(gt_block['T_w_c'], dtype=np.float64)
        gt_world = transform_points(gt_pts3d.reshape(-1, 3), T_w_c_gt)

        pred_mask_flat = np.asarray(rd['pred']['depth_mask']).reshape(-1)
        gt_mask_flat = gt_mask_2d.reshape(-1)
        both_valid = pred_mask_flat & gt_mask_flat
        if not both_valid.any():
            continue

        scale = float(rd['pred'].get('similarity_scale', 1.0))
        pred_world = transform_points(
            (rd['pred']['pts3d'] * scale).reshape(-1, 3),
            np.asarray(rd['pred']['T_w_c'], dtype=np.float64),
        )

        pred_conf = rd['pred'].get('depth_confidence')
        if pred_conf is None:
            pred_conf = pred_mask_flat.astype(np.float32)
        pred_conf_flat = np.asarray(pred_conf, dtype=np.float64).reshape(-1)
        gt_conf_flat = np.ones_like(pred_conf_flat)

        src_pts_chunks.append(pred_world[both_valid])
        tgt_pts_chunks.append(gt_world[both_valid])
        src_conf_chunks.append(pred_conf_flat[both_valid])
        tgt_conf_chunks.append(gt_conf_flat[both_valid])

    if len(src_pts_chunks) == 0:
        return

    src_pts = np.concatenate(src_pts_chunks, axis=0)
    tgt_pts = np.concatenate(tgt_pts_chunks, axis=0)
    src_conf = np.concatenate(src_conf_chunks, axis=0)
    tgt_conf = np.concatenate(tgt_conf_chunks, axis=0)

    median_conf = float(np.median(src_conf))
    conf_threshold = 0.1 * median_conf if median_conf > 0 else 0.0

    s, R, t = weighted_align_point_maps(
        src_pts, src_conf, tgt_pts, tgt_conf,
        conf_threshold=conf_threshold,
    )

    for rd in result_list_per_sub_scene:
        prev_scale = float(rd['pred'].get('similarity_scale', 1.0))
        T = np.asarray(rd['pred']['T_w_c'], dtype=np.float64)
        T_new = np.eye(4, dtype=np.float64)
        T_new[:3, :3] = R @ T[:3, :3]
        T_new[:3, 3] = s * (R @ T[:3, 3]) + t
        rd['pred']['T_w_c_before_similarity'] = rd['pred']['T_w_c']
        rd['pred']['T_w_c'] = T_new
        rd['pred']['similarity_scale'] = s * prev_scale
        rd['pred']['similarity_R'] = R
        rd['pred']['similarity_t'] = t


def refine_similarity_sub_scene(result_list_per_sub_scene, similarity_refine_use_pose_init):
    source_world_all = []
    target_world_all = []

    if len(result_list_per_sub_scene) == 0:
        return

    for result_dict in result_list_per_sub_scene:
        source_camera, _ = pred_points_from_result(result_dict)
        if source_camera.shape[0] == 0:
            continue
        target_camera = gt_pointcloud_from_result(result_dict)
        if target_camera.shape[0] == 0:
            continue
        source_world = transform_points(source_camera, result_dict['pred']['T_w_c'])
        target_world = transform_points(target_camera, result_dict['GT']['T_w_c'])

        source_world_all.append(source_world)
        target_world_all.append(target_world)

    if len(source_world_all) == 0:
        return

    source_world_all = np.concatenate(source_world_all, axis=0)
    target_world_all = np.concatenate(target_world_all, axis=0)
    init_transform = None
    if similarity_refine_use_pose_init:
        pose_alignment_result = estimate_pose_alignment_transform(result_list_per_sub_scene)
        if pose_alignment_result is not None:
            init_transform = pose_alignment_result[0]
    alignment_result = estimate_pointcloud_alignment_transform(
        source_world_all,
        target_world_all,
        init_transform=init_transform,
    )
    if alignment_result is None:
        return
    (
        transform_source_world_to_target_world,
        similarity_scale,
        similarity_R,
        similarity_t,
    ) = alignment_result

    for result_dict in result_list_per_sub_scene:
        pred_T_w_c = np.asarray(result_dict['pred']['T_w_c'], dtype=np.float64)
        aligned_T_w_c = pred_T_w_c.copy()
        aligned_T_w_c[:3, :3] = similarity_R @ pred_T_w_c[:3, :3]
        aligned_T_w_c[:3, 3] = (
            similarity_scale * similarity_R @ pred_T_w_c[:3, 3]
            + similarity_t
        )
        aligned_T_w_c[3] = np.array([0.0, 0.0, 0.0, 1.0])

        result_dict['pred']['T_w_c_before_similarity'] = result_dict['pred']['T_w_c']
        result_dict['pred']['T_w_c'] = aligned_T_w_c
        result_dict['pred']['similarity_scale'] = similarity_scale
        result_dict['pred']['similarity_R'] = similarity_R
        result_dict['pred']['similarity_t'] = similarity_t
        result_dict['pred']['sim3_T_w_c'] = (
            transform_source_world_to_target_world
            @ pred_T_w_c
        )
        result_dict['pred']['similarity_transform'] = transform_source_world_to_target_world



def consistency_check(result_list_per_sub_scene, tol_reproject_err = 1.414, num_consistency_num = 2):


    for i, result_dict_i in enumerate(result_list_per_sub_scene):

        T_w_ci = result_dict_i['pred']['T_w_c']
        scale_i = result_dict_i['pred'].get('similarity_scale', 1.0)
        pts3d_i = result_dict_i['pred']['pts3d'] * scale_i
        H, W, C = result_dict_i['pred']['depth'].shape


        consistency_check_matrix = np.zeros((H,W), dtype=np.int64)

        for j, result_dict_j in enumerate(result_list_per_sub_scene):

            if result_dict_i['basic']['sample_idx'] == result_dict_j['basic']['sample_idx']: 
                continue

            T_w_cj = result_dict_j['pred']['T_w_c']
            T_cj_ci = np.linalg.inv(T_w_cj) @ T_w_ci
            T_cj_ci_Trans = T_cj_ci.T
            reproject_pcj_h = pts3d_i @ T_cj_ci_Trans[:-1, :] + T_cj_ci_Trans[-1:, :]
            reproject_pcj = reproject_pcj_h[:, :, :3]
            reproject_depth_j = reproject_pcj_h[:, :, 2]
            K_matrix = result_dict_j['pred']['intrinsics']
            K_matrix_Trans = K_matrix.T
            reproject_pcj_in_norm_plane = reproject_pcj @ K_matrix_Trans
            safe_z_j = np.where(reproject_depth_j > 1e-6, reproject_depth_j, 1.0)[:, :, None]
            reproject_uv1 = (reproject_pcj_in_norm_plane / safe_z_j).round().astype(np.int64)
            reproject_u = reproject_uv1[:, :, 0]
            reproject_v = reproject_uv1[:, :, 1]
            mask_for_i = (
                np.isfinite(reproject_depth_j)
                & (reproject_depth_j > 0)
                & (0 <= reproject_u) & (reproject_u < W)
                & (0 <= reproject_v) & (reproject_v < H)
            )
            if not mask_for_i.any():
                continue

            # Round-trip UV reprojection error: take frame j's own 3D point at the
            # reprojected pixel, send it back to camera i, project, and compare to
            # the original (u_i, v_i).
            mask_for_j = reproject_v[mask_for_i], reproject_u[mask_for_i]
            scale_j = result_dict_j['pred'].get('similarity_scale', 1.0)
            pts3d_j_scaled = result_dict_j['pred']['pts3d'] * scale_j  # (H, W, 3) in camera_j
            P_cj_from_j = pts3d_j_scaled[mask_for_j]  # (N, 3)

            T_ci_cj = np.linalg.inv(T_cj_ci)
            P_ci_back = P_cj_from_j @ T_ci_cj[:3, :3].T + T_ci_cj[:3, 3]

            K_i = result_dict_i['pred']['intrinsics']
            uv_back_h = P_ci_back @ K_i.T
            z_back = uv_back_h[:, 2]
            valid_z = z_back > 1e-6
            safe_z_back = np.where(valid_z, z_back, 1.0)
            uv_back = uv_back_h[:, :2] / safe_z_back[:, None]

            v_idx, u_idx = np.nonzero(mask_for_i)
            uv_orig = np.stack((u_idx, v_idx), axis=-1).astype(np.float64)  # (u, v)

            uv_err = np.linalg.norm(uv_back - uv_orig, axis=-1)
            acceptable_mask = valid_z & np.isfinite(uv_err) & (uv_err < tol_reproject_err)
            consistency_check_matrix[mask_for_i] += acceptable_mask.astype(np.int64)

        result_dict_i['pred']['depth_mask'] = result_dict_i['pred']['depth_mask'] & (consistency_check_matrix >= num_consistency_num)
