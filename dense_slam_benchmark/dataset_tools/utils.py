from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import cv2
from decimal import Decimal, getcontext
from dense_slam_benchmark.dataset_tools import undistort


getcontext().prec = 30  # enough for seconds + 9 fractional digits


def single_depths2colors(depth, min_depth, max_depth):
    depth = np.asarray(depth, dtype=np.float32)

    # clip and normalize to [0, 255]
    depth_clipped = np.clip(depth, min_depth, max_depth)
    normalized = 255 * (depth_clipped - min_depth) / (max_depth - min_depth)
    normalized = (255 - normalized).astype(np.uint8)

    # matplotlib colormap expects values in [0, 1]
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    colors_rgb = (cmap(normalized / 255.0)[..., :3] * 255).astype(np.uint8)

    # RGB -> BGR
    colors_bgr = colors_rgb[..., ::-1]

    return colors_bgr

def single_depth2color(depth, min_depth, max_depth):

    depth_clipped = np.clip(depth, min_depth, max_depth)
    normalized = int(255 * (depth_clipped - min_depth) / (max_depth - min_depth))    
    normalized = np.uint8(255 - normalized)  # Flip so small depth is red

    # Apply colormap on a 1x1 image
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    color = np.array(cmap(normalized)[:3]) * 255
    color = color.astype(np.uint8)
    # Extract the color as (B, G, R)
    return (int(color[2]), int(color[1]), int(color[0]))


def depth2color(
    depth,
    cmap_name="Spectral_r",
    invalid_color=(0, 0, 0),
    max_depth = 10,
    min_depth = 0,
):
    """
    Convert inverse depth to depth using:
        mask = inv_depth > 0
        depth = 0
        depth[mask] = 1.0 / inv_depth[mask]

    Then colorize and save as an RGB image.

    Args:
        inv_depth: numpy array of shape (H, W) or (1, H, W)
        save_path: output image path, e.g. "depth_color.png"
        cmap_name: matplotlib colormap name
        invalid_color: RGB tuple for invalid pixels
    """

    mask = (depth <= max_depth) & (depth >= min_depth) & (depth > 0)
    inv_depth = np.zeros_like(depth) 
    inv_depth[mask] = 1.0 / depth[mask]

    vmin = inv_depth[mask].min()
    vmax = inv_depth[mask].max()

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    colored = cmap(norm(inv_depth))            # RGBA in [0, 1]
    rgb = (colored[..., :3] * 255).astype(np.uint8)
    bgr = rgb[..., ::-1].copy()

    bgr[~mask] = np.array(invalid_color, dtype=np.uint8)

    return bgr

def voxel_downsample_np(points, voxel_size=0.05):
    """
    points: (N, 3) or (N, C), first 3 columns must be xyz
    returns: downsampled array with one averaged point per voxel
    """
    xyz = points[:, :3]
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)

    _, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    num_voxels = inverse.max() + 1

    counts = np.bincount(inverse)

    down = np.zeros((num_voxels, points.shape[1]), dtype=np.float64)
    for j in range(points.shape[1]):
        down[:, j] = np.bincount(inverse, weights=points[:, j]) / counts

    return down


def depth_range_by_ratio(depth, keep=0.98):
    """
    Return the central depth range containing `keep` fraction of valid depth values.

    Args:
        depth: numpy array
        keep: float in (0, 1], e.g. 0.95 means central 95%

    Returns:
        d_low, d_high
    """
    depth = np.asarray(depth, dtype=np.float32)

    if not (0 < keep <= 1):
        raise ValueError(f"`keep` must be in (0, 1], got {keep}")

    valid = depth[np.isfinite(depth) & (depth > 0)]

    if valid.size == 0:
        raise ValueError("No valid depth pixels found")

    tail = (1.0 - keep) / 2.0
    q_low = 100.0 * tail
    q_high = 100.0 * (1.0 - tail)

    d_low, d_high = np.percentile(valid, [q_low, q_high])
    return d_low, d_high


def save_depth_histogram(
    depth,
    save_path="depth_hist.png",
    min_depth=0.0,
    max_depth=10.0,
):
    depth = np.asarray(depth, dtype=np.float32)

    mask = np.isfinite(depth) & (depth > 0) & (depth >= min_depth) & (depth <= max_depth)
    valid_depth = depth[mask].ravel()

    if valid_depth.size == 0:
        raise ValueError("No valid depth pixels found")

    # Let NumPy decide the bin edges automatically
    bin_edges = np.histogram_bin_edges(valid_depth, bins="auto")

    # Compute histogram explicitly
    counts, _ = np.histogram(valid_depth, bins=bin_edges)

    plt.figure(figsize=(6, 4))
    plt.stairs(counts, bin_edges)
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.title("Depth Histogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def isInImage(u, v, z, width, height):
    if z <= 0 or u < 0 or u > width - 1 or v < 0 or v > height - 1:
        return False
    else:
        return True

def pose_to_T(x, y, z, qx, qy, qz, qw):
    R = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [x, y, z]
    return T

def T_to_pose(T: np.ndarray):
    """
    Convert a 4x4 homogeneous transform to translation + quaternion.

    Args:
        T: (4,4) array-like homogeneous transform.

    Returns:
        t: (3,) translation vector [x, y, z]
        q: (4,) quaternion [q_x, q_y, q_z, q_w]  (SciPy order: x,y,z,w)
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be shape (4,4), got {T.shape}")

    t = T[:3, 3].copy()
    q = Rot.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
    return t, q

def getSynchronizedSensorIdx(ref_ts, sensor_names, ts_tol):

    # whether it can find out the synchronized signal or not, it will return the idx that is closest to the ref_ts in timestamp
    

    synchronized_idx = -1
    minimum_ts_diff = float("inf")
    for i, sensor_name in enumerate(sensor_names):
        sensor_ts = sensor_name.split(".")[0]
        sensor_ts = int(sensor_ts) / 1e9
        ts_diff = abs(ref_ts - sensor_ts)
        if ts_diff < minimum_ts_diff:
            minimum_ts_diff = ts_diff
            synchronized_idx = i
    
    if minimum_ts_diff <= ts_tol :
        return True, synchronized_idx
    else:
        return False, synchronized_idx

def getSensorIdxWithClosestTimeStamp(ref_ts, sensor_names):

    # whether it can find out the synchronized signal or not, it will return the idx that is closest to the ref_ts in timestamp
    

    idx_with_closest_timestamp = -1
    minimum_ts_diff = float("inf")
    for i, sensor_name in enumerate(sensor_names):
        sensor_ts = sensor_name.split(".")[0]
        sensor_ts = int(sensor_ts) / 1e9
        ts_diff = abs(ref_ts - sensor_ts)
        if ts_diff < minimum_ts_diff:
            minimum_ts_diff = ts_diff
            idx_with_closest_timestamp = i
    
    return idx_with_closest_timestamp


def getSynchronizedPoseIdx(ref_ts, samples, ts_tol):

    synchronized_idx = -1
    minimum_ts_diff = float("inf")
    for i, sample in enumerate(samples):
        sample_ts = sample['ts']
        ts_diff = abs(ref_ts - sample_ts)
        if ts_diff < minimum_ts_diff:
            minimum_ts_diff = ts_diff
            synchronized_idx = i
    
    if minimum_ts_diff <= ts_tol :
        return True, synchronized_idx
    else:
        return False, synchronized_idx


def timestamp_str_to_float(ts_str):

    return float(ts_str[:10]+"."+ts_str[10:])

def calculateUndistortedRemap(
    distortion_model,
    resolution,
    intrinsics,
    distortion,
    target_resolution=None,
    target_intrinsics=None,
):

    image_w, image_h = resolution
    fx, fy, cx, cy = intrinsics
    distortion = np.asarray(distortion, dtype=np.float32).reshape(-1)
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]], dtype=np.float32
    )

    if distortion_model == 'radtan':
        if distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError(
                f"radtan expects 4/5/8/12/14 coefficients, got {distortion.size}: {distortion.tolist()}"
            )

        D = distortion
        if target_intrinsics is None:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (image_w, image_h), alpha=0.0)
        else:
            newK = np.array(
                [
                    [target_intrinsics[0], 0, target_intrinsics[2]],
                    [0, target_intrinsics[1], target_intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        target_size = tuple(target_resolution) if target_resolution is not None else (image_w, image_h)
        remap1, remap2 = cv2.initUndistortRectifyMap(
            K, D, R=None, newCameraMatrix=newK, size=target_size, m1type=cv2.CV_32FC1
        )

    elif distortion_model == 'equidistant':
        if distortion.size != 4:
            raise ValueError(
                f"equidistant expects 4 coefficients, got {distortion.size}: {distortion.tolist()}"
            )

        D = distortion
        if target_intrinsics is None:
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (image_w, image_h), np.eye(3), balance=0.0, new_size=(image_w, image_h), fov_scale=1.0
            )
        else:
            newK = np.array(
                [
                    [target_intrinsics[0], 0, target_intrinsics[2]],
                    [0, target_intrinsics[1], target_intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        target_size = tuple(target_resolution) if target_resolution is not None else (image_w, image_h)
        remap1, remap2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, target_size, m1type=cv2.CV_32FC1
        )

    elif distortion_model == 'thin_prism_fisheye':
        if distortion.size != 8:
            raise ValueError(
                f"thin_prism_fisheye expects 8 coefficients [k1, k2, p1, p2, k3, k4, sx1, sy1], "
                f"got {distortion.size}: {distortion.tolist()}"
            )

        try:
            newK, remap1, remap2 = undistort.create_colmap_thin_prism_fisheye_undistort_map(
                resolution=(image_w, image_h),
                intrinsics=(fx, fy, cx, cy),
                distortion_coeffs=distortion,
                new_intrinsics=target_intrinsics,
                target_resolution=target_resolution,
            )
        except ImportError:
            newK, remap1, remap2 = undistort.create_thin_prism_fisheye_undistort_map(
                resolution=(image_w, image_h),
                intrinsics=(fx, fy, cx, cy),
                distortion_coeffs=distortion,
                new_intrinsics=target_intrinsics,
                target_resolution=target_resolution,
            )

    else:
        raise ValueError(f"Unsupported distortion model: {distortion_model}")


    return newK, remap1, remap2



def undistortedDepth2Pointcloud(depth_image, intrinsics):

    fx, fy, cx, cy = intrinsics

    pc_h_list = []
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            z =  depth_image[v, u]
            if z <= 1e-3 or z > 1e3:
                # print(f"{v},{u} invalid z:{z}")
                continue                            
            x = (u - cx)*z/fx
            y = (v - cy)*z/fy
            pc_h_list.append(np.array([x,y,z,1], dtype='f4'))

    pc_hs = np.vstack(pc_h_list, dtype='f4')

    return pc_hs
