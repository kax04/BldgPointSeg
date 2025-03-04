import numpy as np
import open3d as o3d
import re
from tqdm import tqdm


# 各平面の中心点を計算する関数
def compute_plane_center(corners):
    return np.mean(np.array(corners), axis=0)

# 視錐台の中心点を計算する関数
def compute_frustum_center(near_corners, far_corners):
    near_center = np.mean(np.array(near_corners), axis=0)
    far_center = np.mean(np.array(far_corners), axis=0)
    return (near_center + far_center) / 2

def create_frustum_mesh(sensor_size, fov, aspect_ratio, near_clip, far_clip, camera_pos, camera_rot):
    # センサーサイズからアスペクト比を計算
    sensor_aspect_ratio = sensor_size[0] / sensor_size[1]

    # Field of view, sensor aspect ratio, and clip planes
    tan_fov = np.tan(np.radians(fov / 2.0))
    half_height_near = near_clip * tan_fov
    half_width_near = half_height_near * sensor_aspect_ratio
    half_height_far = far_clip * tan_fov
    half_width_far = half_height_far * sensor_aspect_ratio

    camera_rotation_radians = np.radians(camera_rot)
    print("camera_rotation_radians: ", camera_rotation_radians)
    # 回転行列を計算
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(camera_rotation_radians)

    # Calculate corners of the frustum
    near_plane_corners = [
        np.array([half_width_near, half_height_near, near_clip]),
        np.array([-half_width_near, half_height_near, near_clip]),
        np.array([-half_width_near, -half_height_near, near_clip]),
        np.array([half_width_near, -half_height_near, near_clip])
    ]
    far_plane_corners = [
        np.array([half_width_far, half_height_far, far_clip]),
        np.array([-half_width_far, half_height_far, far_clip]),
        np.array([-half_width_far, -half_height_far, far_clip]),
        np.array([half_width_far, -half_height_far, far_clip])
    ]

    # 回転を適用してカメラ位置を加算
    corners = [camera_pos + np.dot(rotation_matrix, corner) for corner in near_plane_corners + far_plane_corners]

    # Create lines between the corners
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
    ]

    # Create a line set for the frustum
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(len(lines))])  # Black color for lines

    return line_set

# カメラパラメータを読み込む関数
def load_camera_parameters(camera_params_file):
    with open(camera_params_file, 'r') as file:
        params = file.read()

    sensor_size = tuple(map(float, re.search(r"Sensor Size: \(([\d.]+), ([\d.]+)\)", params).groups()))
    focal_length = float(re.search(r"Focal Length: ([\d.]+)", params).group(1))
    fov = float(re.search(r"Field of View: ([\d.]+)", params).group(1))
    aspect_ratio = float(re.search(r"Aspect Ratio: ([\d.]+)", params).group(1))
    near_clip = float(re.search(r"Near Clip Plane: ([\d.]+)", params).group(1))
    far_clip = float(re.search(r"Far Clip Plane: ([\d.]+)", params).group(1))
    position = tuple(map(float, re.search(r"Position: \(([\d.]+), ([\d.]+), ([\d.]+)\)", params).groups()))
    position = (-position[0], position[1], position[2])  # X座標をマイナスにする
    rotation = tuple(map(float, re.search(r"Rotation: \(([\d.]+), ([\d.]+), ([\d.]+)\)", params).groups()))

    return sensor_size, focal_length, fov, aspect_ratio, near_clip, far_clip, position, rotation

# 各平面のメッシュを作成する関数
def create_mesh_from_corners(corners, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])  # 2つの三角形で四角形を形成
    mesh.paint_uniform_color(color)
    return mesh


# Load camera parameters
sensor_size, focal_length, fov, aspect_ratio, near_clip, far_clip, camera_pos, camera_rot = load_camera_parameters("./image/depth/per2/hallway.txt")

# Load point cloud
pcd = o3d.io.read_point_cloud("./image/depth/per2/hallway.ply")

# Rotate the point cloud
rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
pcd.rotate(rotation_matrix, center=(0, 0, 0))

# 点群を上下反転させる
flip_transform = np.array([[-1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
pcd.transform(flip_transform)


# Create frustum mesh
frustum_mesh = create_frustum_mesh(sensor_size, fov, aspect_ratio, near_clip, far_clip, camera_pos, camera_rot)

# LineSetのpoints属性から視錐台のコーナーを取得
corners = np.asarray(frustum_mesh.points)




# 各平面の頂点を定義
near_plane_corners = corners[:4]
far_plane_corners = corners[4:]
left_plane_corners = [corners[i] for i in [1, 5, 6, 2]]
right_plane_corners = [corners[i] for i in [0, 4, 7, 3]]
top_plane_corners = [corners[i] for i in [0, 1, 4, 5]]
bottom_plane_corners = [corners[i] for i in [2, 3, 6, 7]]

# 各平面の中心点を計算
near_center = compute_plane_center(near_plane_corners)
far_center = compute_plane_center(far_plane_corners)
left_center = compute_plane_center(left_plane_corners)
right_center = compute_plane_center(right_plane_corners)
top_center = compute_plane_center(top_plane_corners)
bottom_center = compute_plane_center(bottom_plane_corners)

# 視錐台の中心点を計算
frustum_center = compute_frustum_center(near_plane_corners, far_plane_corners)

# 各平面のメッシュを作成
near_plane_mesh = create_mesh_from_corners(near_plane_corners, [1, 0, 0])  # 赤色
far_plane_mesh = create_mesh_from_corners(far_plane_corners, [0, 1, 0])  # 緑色
left_plane_mesh = create_mesh_from_corners(left_plane_corners, [0, 0, 1])  # 青色
right_plane_mesh = create_mesh_from_corners(right_plane_corners, [1, 1, 0])  # 黄色
top_plane_mesh = create_mesh_from_corners(top_plane_corners, [1, 0, 1])  # マゼンタ
bottom_plane_mesh = create_mesh_from_corners(bottom_plane_corners, [0, 1, 1])  # シアン


# 作成した平面のメッシュをリストに追加
plane_meshes = [
    near_plane_mesh,
    far_plane_mesh,
    left_plane_mesh,
    right_plane_mesh,
    top_plane_mesh,
    bottom_plane_mesh
]
# Create camera coordinate frame
camera_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=camera_pos)
print("camera_pos:", camera_pos)

# Create world coordinate frame
world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])


# 視錐台の中心点を視覚化するための小さな球を作成
center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
center_sphere.translate(frustum_center)
center_sphere.paint_uniform_color([1, 0, 0])  # 赤色

# 点群のポイントを取得
pcd_points = np.asarray(pcd.points)

# 点群のz軸が視錐台のnear平面より大きく、far平面より小さいものだけを残す
# near_clip_z = near_plane_corners[0][2]  # near平面のz軸座標を取得
# far_clip_z = far_plane_corners[0][2]    # far平面のz軸座標を取得

# 世界座標系に変換された頂点からZ座標を取得し、near平面の最小値とfar平面の最大値を取得
near_clip_z = min(corner[2] for corner in near_plane_corners)
far_clip_z = max(corner[2] for corner in far_plane_corners)

print("near_clip_z (min):", near_clip_z)
print("far_clip_z (max):", far_clip_z)



# 条件に合う点だけをフィルタリング
filtered_indices = np.where((pcd_points[:, 2] > near_clip_z) & (pcd_points[:, 2] < far_clip_z))[0]
filtered_pcd = pcd.select_by_index(filtered_indices)

# フィルタリングした点群を表示
# o3d.visualization.draw_geometries([filtered_pcd])

# 各平面の法線ベクトルを計算
def compute_plane_normal(corners):
    p1, p2, p3 = corners[:3]
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    return normal

# 点が平面の内側にあるかどうかを判定する関数
def is_point_inside_plane(point, plane_corners, plane_normal):
    plane_point = plane_corners[0]  # 平面上の任意の点
    point_vector = point - plane_point
    return np.dot(plane_normal, point_vector) <= 0

# 各平面の法線ベクトルを計算し、必要に応じて反転
near_plane_normal = compute_plane_normal(near_plane_corners) * -1
far_plane_normal = compute_plane_normal(far_plane_corners)
left_plane_normal = compute_plane_normal(left_plane_corners) * -1
right_plane_normal = compute_plane_normal(right_plane_corners)
top_plane_normal = compute_plane_normal(top_plane_corners)
bottom_plane_normal = compute_plane_normal(bottom_plane_corners)

# 点がすべての平面の内側にあるかどうかを判定する関数
def is_point_inside_all_planes(point, plane_corners_normals):
    return all(is_point_inside_plane(point, corners, normal) for corners, normal in plane_corners_normals)

# 各平面のコーナーと法線ベクトルのペアをリストに格納
plane_corners_normals = [
    (near_plane_corners, near_plane_normal),
    (far_plane_corners, far_plane_normal),
    (left_plane_corners, left_plane_normal),
    (right_plane_corners, right_plane_normal),
    (top_plane_corners, top_plane_normal),
    (bottom_plane_corners, bottom_plane_normal)
]

# フィルタリングされた点群のインデックスから、すべての平面の内側にある点のインデックスを抽出
inside_all_planes_indices = [i for i in tqdm(filtered_indices, desc="Checking points inside planes") if is_point_inside_all_planes(pcd_points[i], plane_corners_normals)]

# 条件に合う点だけを選択
inside_all_planes_pcd = pcd.select_by_index(inside_all_planes_indices)

# inside_all_planes_pcdをPLYファイルとして保存
o3d.io.write_point_cloud("hallway_filtered.ply", inside_all_planes_pcd, write_ascii=True)

# o3d.visualization.draw_geometries([pcd,
#                                    camera_coordinate_frame,
#                                    frustum_mesh,
#                                    ])

# 視覚化
o3d.visualization.draw_geometries([inside_all_planes_pcd,
                                   camera_coordinate_frame,
                                   world_coordinate_frame,
                                   frustum_mesh,
                                   *plane_meshes,
                                   center_sphere])
