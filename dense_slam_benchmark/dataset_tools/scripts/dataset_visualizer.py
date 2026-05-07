import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.path.entities import Line


DEFAULT_DATASET_DIR = Path(
    "/mnt/lboro_nas/personal/Zhipeng/wai_data/BotanicGarden/"
    "BotanicGarden_1018_00_64"
)
DEFAULT_POINTCLOUD_NAME = "livox_lidar_c0.pcd"
# DEFAULT_POSE_FILES = [
#     DEFAULT_DATASET_DIR
#     / "left_rgb"
#     / "liosam_T_left_rgb_first_sample_from_cam0.txt",
#     DEFAULT_DATASET_DIR
#     / "right_rgb"
#     / "liosam_T_left_rgb_first_sample_from_cam1.txt",
# ]
# DEFAULT_POSE_NAMES = ["left_rgb", "right_rgb"]

DEFAULT_POSE_FILES = [
    DEFAULT_DATASET_DIR
    / "left_rgb"
    / "liosam_T_left_rgb_first_sample_from_cam0.txt"
]
DEFAULT_POSE_NAMES = ["left_rgb"]

POSE_COLORS = [
    [255, 80, 40, 255],
    [40, 180, 255, 255],
    [90, 230, 90, 255],
    [230, 180, 40, 255],
    [190, 110, 255, 255],
    [255, 110, 190, 255],
    [80, 230, 220, 255],
    [220, 220, 220, 255],
]


def quat_xyzw_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion has zero norm.")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_T_w_c(x, y, z, qx, qy, qz, qw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat([qx, qy, qz, qw])
    T[:3, 3] = [x, y, z]
    return T


def read_pose_file(path):
    poses = []
    timestamps = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(
                    f"{path}:{line_num} expected 8 columns: "
                    "timestamp/index x y z qx qy qz qw"
                )
            timestamps.append(parts[0])
            x, y, z, qx, qy, qz, qw = map(float, parts[1:])
            poses.append(pose_to_T_w_c(x, y, z, qx, qy, qz, qw))
    if len(poses) == 0:
        raise ValueError(f"No poses found in {path}")
    return timestamps, np.stack(poses, axis=0)


def parse_pcd_header(path):
    header = {}
    header_lines = []
    offset = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line == b"":
                raise ValueError(f"PCD header in {path} ended before DATA line.")
            offset += len(line)
            text = line.decode("utf-8", errors="replace").strip()
            header_lines.append(text)
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
    return header, header_lines, offset


def pcd_dtype(header):
    fields = header["FIELDS"]
    sizes = [int(v) for v in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]

    dtype_fields = []
    for field, size, type_code, count in zip(fields, sizes, types, counts):
        if type_code == "F" and size == 4:
            dtype = np.float32
        elif type_code == "F" and size == 8:
            dtype = np.float64
        elif type_code == "U" and size == 1:
            dtype = np.uint8
        elif type_code == "U" and size == 2:
            dtype = np.uint16
        elif type_code == "U" and size == 4:
            dtype = np.uint32
        elif type_code == "I" and size == 1:
            dtype = np.int8
        elif type_code == "I" and size == 2:
            dtype = np.int16
        elif type_code == "I" and size == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported PCD field type: {field} {type_code}{size}")

        if count == 1:
            dtype_fields.append((field, dtype))
        else:
            dtype_fields.append((field, dtype, (count,)))
    return np.dtype(dtype_fields)


def decode_pcd_rgb(rgb_values):
    rgb_values = np.asarray(rgb_values)
    if rgb_values.dtype.kind == "f":
        packed = rgb_values.astype(np.float32, copy=False).view(np.uint32)
    else:
        packed = rgb_values.astype(np.uint32, copy=False)
    r = ((packed >> 16) & 255).astype(np.uint8)
    g = ((packed >> 8) & 255).astype(np.uint8)
    b = (packed & 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


def sample_indices(num_points, max_points, seed):
    if max_points is None or num_points <= max_points:
        return np.arange(num_points)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=max_points, replace=False))


def load_pcd_as_points(path, max_points=2_000_000, seed=0):
    path = Path(path)
    header, _, data_offset = parse_pcd_header(path)
    data_mode = header["DATA"][0].lower()
    fields = header["FIELDS"]
    num_points = int(header.get("POINTS", [header.get("WIDTH", [0])[0]])[0])
    indices = sample_indices(num_points, max_points, seed)

    if data_mode == "binary":
        data = np.memmap(
            path,
            dtype=pcd_dtype(header),
            mode="r",
            offset=data_offset,
            shape=(num_points,),
        )
        sampled = data[indices]
        points = np.column_stack([sampled["x"], sampled["y"], sampled["z"]]).astype(
            np.float32
        )
        colors = decode_pcd_rgb(sampled["rgb"]) if "rgb" in fields else None
    elif data_mode == "ascii":
        data = np.loadtxt(path, comments="#", skiprows=len(parse_pcd_header(path)[1]))
        field_to_idx = {field: idx for idx, field in enumerate(fields)}
        sampled = data[indices]
        points = sampled[:, [field_to_idx["x"], field_to_idx["y"], field_to_idx["z"]]]
        colors = (
            decode_pcd_rgb(sampled[:, field_to_idx["rgb"]])
            if "rgb" in field_to_idx
            else None
        )
    else:
        raise ValueError(f"Unsupported PCD DATA mode '{data_mode}'.")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is None:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    else:
        colors = colors[finite]
    return points, colors


def make_point_cloud(points, colors):
    return trimesh.points.PointCloud(points, colors=colors)


def make_path(vertices, edges, color):
    entities = [Line(edge) for edge in edges]
    path = trimesh.path.Path3D(entities=entities, vertices=np.asarray(vertices))
    rgba = np.asarray(color, dtype=np.uint8)
    for entity in path.entities:
        entity.color = rgba
    return path


def transform_points(T, points):
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def make_camera_marker(T_w_c, scale, color):
    half_w = scale * 0.6
    half_h = scale * 0.4
    depth = scale
    camera_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )
    vertices = transform_points(T_w_c, camera_points)
    edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]
    return make_path(vertices, edges, color)


def make_pose_axes(T_w_c, scale):
    origin = T_w_c[:3, 3]
    x_axis = origin + T_w_c[:3, 0] * scale
    y_axis = origin + T_w_c[:3, 1] * scale
    z_axis = origin + T_w_c[:3, 2] * scale
    return [
        make_path([origin, x_axis], [[0, 1]], [255, 0, 0, 255]),
        make_path([origin, y_axis], [[0, 1]], [0, 255, 0, 255]),
        make_path([origin, z_axis], [[0, 1]], [0, 80, 255, 255]),
    ]


def make_trajectory_path(poses, color):
    centers = poses[:, :3, 3]
    if centers.shape[0] < 2:
        return None
    edges = [[i, i + 1] for i in range(centers.shape[0] - 1)]
    return make_path(centers, edges, color)


def add_pose_sequence(
    scene,
    name,
    poses,
    color,
    pose_stride,
    frustum_scale,
    axis_scale,
):
    trajectory = make_trajectory_path(poses, color)
    if trajectory is not None:
        scene.add_geometry(trajectory, node_name=f"{name}_trajectory")

    for pose_idx in range(0, len(poses), pose_stride):
        scene.add_geometry(
            make_camera_marker(poses[pose_idx], frustum_scale, color),
            node_name=f"{name}_frustum_{pose_idx:06d}",
        )
        for axis_idx, axis in enumerate(make_pose_axes(poses[pose_idx], axis_scale)):
            scene.add_geometry(
                axis,
                node_name=f"{name}_axis_{pose_idx:06d}_{axis_idx}",
            )


def default_pose_files_for_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if dataset_dir == DEFAULT_DATASET_DIR:
        return DEFAULT_POSE_FILES
    return [
        dataset_dir / "left_rgb" / "liosam_T_left_rgb_first_sample_from_cam0.txt",
        dataset_dir / "right_rgb" / "liosam_T_left_rgb_first_sample_from_cam1.txt",
    ]


def default_pose_names_for_files(pose_files):
    names = []
    used = set()
    for path in pose_files:
        path = Path(path)
        name = path.parent.name if path.parent.name else path.stem
        if name in used:
            name = path.stem
        original = name
        suffix = 1
        while name in used:
            name = f"{original}_{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)
    return names


def build_scene(args):
    points, colors = load_pcd_as_points(
        args.pointcloud,
        max_points=args.max_points,
        seed=args.seed,
    )

    scene = trimesh.Scene()
    scene.add_geometry(make_point_cloud(points, colors), node_name="scene_pointcloud")

    for pose_idx, (pose_name, pose_path) in enumerate(zip(args.pose_names, args.poses)):
        _, poses = read_pose_file(pose_path)
        add_pose_sequence(
            scene,
            pose_name,
            poses,
            POSE_COLORS[pose_idx % len(POSE_COLORS)],
            args.pose_stride,
            args.frustum_scale,
            args.axis_scale,
        )

    return scene


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a dataset point cloud and camera pose trajectories with trimesh."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Dataset directory containing .pcd and pose folders.",
    )
    parser.add_argument(
        "--pointcloud",
        type=Path,
        default=DEFAULT_POINTCLOUD_NAME,
        help=(
            "Scene PCD path. Relative paths are resolved inside --dataset_dir. "
            f"Defaults to <dataset_dir>/{DEFAULT_POINTCLOUD_NAME}."
        ),
    )
    parser.add_argument(
        "--poses",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more pose txt files. Defaults to the BotanicGarden "
            "left_rgb and right_rgb pose files."
        ),
    )
    parser.add_argument(
        "--pose_names",
        nargs="+",
        default=None,
        help="Optional display names for --poses. Must have the same length as --poses.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=2_000_000,
        help="Maximum scene points to display. Use a smaller value if rendering is slow.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pose_stride",
        type=int,
        default=1,
        help="Draw one camera frustum every N poses.",
    )
    parser.add_argument("--frustum_scale", type=float, default=0.4)
    parser.add_argument("--axis_scale", type=float, default=0.25)
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to export the trimesh scene, e.g. /tmp/botanic_scene.glb.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Build/export the scene without opening the interactive viewer.",
    )
    args = parser.parse_args()

    if not args.pointcloud.is_absolute():
        args.pointcloud = args.dataset_dir / args.pointcloud
    args.poses = args.poses or default_pose_files_for_dataset(args.dataset_dir)
    if args.pose_names is None:
        if args.poses == DEFAULT_POSE_FILES:
            args.pose_names = DEFAULT_POSE_NAMES
        else:
            args.pose_names = default_pose_names_for_files(args.poses)
    if len(args.pose_names) != len(args.poses):
        raise ValueError(
            f"--pose_names has {len(args.pose_names)} entries, "
            f"but --poses has {len(args.poses)} files."
        )
    return args



def main():
    args = parse_args()
    scene = build_scene(args)

    if args.export is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        scene.export(args.export)
        print(f"Exported scene to {args.export}")

    if not args.no_show:
        try:
            scene.show()
        except ImportError as exc:
            if "pyglet<2" in str(exc):
                raise ImportError(
                    "trimesh interactive viewing requires pyglet<2. "
                    "Install/update the project dependencies, or run: "
                    "pip install 'pyglet<2'. You can still export without a "
                    "viewer by passing --no_show --export /path/to/scene.glb."
                ) from exc
            raise


# conda run -n benchmark-hloc python dense_slam_benchmark/dataset_tools/scripts/dataset_visualizer.py \
#   --pointcloud /mnt/lboro_nas/personal/Zhipeng/wai_data/BotanicGarden/BotanicGarden_1018_00_64/livox_lidar_c8.pcd \
#   --poses \
#     /path/to/cam0_pose.txt \
#     /path/to/cam1_pose.txt \
#     /path/to/cam2_pose.txt \
#   --pose_names cam0 cam1 cam2 \
#   --max_points 500000 \
#   --pose_stride 20

if __name__ == "__main__":
    main()
