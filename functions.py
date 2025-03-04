#import sys;print(sys.prefix);print(sys.path)


import os

import cv2
import numpy as np
import polars as pl
from utils_our import pred_lines
import open3d as o3d
import re
from tqdm import tqdm
import math
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


from line_classification import calculate_rad_per1, calculate_rad_per2

# CATEGORIES = ["per1", "per2", "per3"]
# CATEGORIES = ["per2"]
# CATEGORIES = ["per1"]


def get_path(input_path, perspective):
    """
    画像が入っているフォルダのパスを取得する関数

    Args:
        input_path : 画像フォルダが入っているパス
    Returns:
        paths : 画像フォルダのパスのリスト
    """

    if perspective==1:
        CATEGORIES = "per1"
    elif perspective==2:
        CATEGORIES = "per2"

    folder_input = os.listdir(input_path)
    new_folder_input = []
    for f in folder_input:
        # if f == CATEGORIES[0] or f == CATEGORIES[1] or f == CATEGORIES[2]:
        if f == CATEGORIES:
            new_folder_input.append(f)
    paths = []
    for folder in new_folder_input:
        folder_path = os.path.join(input_path, folder)
        image_paths = []
        for image in os.listdir(folder_path):
            if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                image_path = os.path.join(folder_path, image)
                image_paths.append(image_path)
        paths.append(image_paths)
    return paths


def get_lines(
    image_path, interpreter, input_details, output_details, num_of_img, num_of_per
):
    """MLSDで線分を検出する関数

    Args:
        image_path (str): 画像のパス
        interpreter : tfliteモデル
        input_details : tfliteモデルの入力
        output_details : tfliteモデルの出力
        num_of_img : 画像の番号
        num_of_per : 透視画像の番号(0,1,2)

    Returns:
        mlsd_lines_sphere : 球面座標系で表された線分のリスト
        mlsd_lines : 画像座標系で表された線分のリスト
        width : 画像の幅
        height : 画像の高さ
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #print(image)
    if np.max(image) <= 1:
        image *= 255

    width = image.shape[1]
    height = image.shape[0]
    scale_w = np.maximum(width, height)
    scale_h = scale_w



    mlsd_lines = pred_lines(
        image_path,
        interpreter,
        input_details,
        output_details,
        num_of_img,
        num_of_per,
        input_shape=[512, 512],
        score_thr=0.1,
        dist_thr=20.0,
    )
    mlsd_lines_sphere = mlsd_lines.copy()

    # 画像の大きさに対して線分を正規化
    mlsd_lines_sphere[:, 0] -= width / 2.0
    mlsd_lines_sphere[:, 1] -= height / 2.0
    mlsd_lines_sphere[:, 2] -= width / 2.0
    mlsd_lines_sphere[:, 3] -= height / 2.0
    mlsd_lines_sphere[:, 0] /= scale_w / 2.0
    mlsd_lines_sphere[:, 1] /= scale_h / 2.0
    mlsd_lines_sphere[:, 2] /= scale_w / 2.0
    mlsd_lines_sphere[:, 3] /= scale_h / 2.0
    mlsd_lines_sphere[:, 1] *= -1
    mlsd_lines_sphere[:, 3] *= -1
    return mlsd_lines_sphere, mlsd_lines, width, height


def create_line_list(mlsd_lines_sphere, mlsd_lines, height, num_of_per, perspective):
    angle, line_class = None, None
    # if num_of_per == 0:
    #     angle, line_class = calculate_rad_per1(mlsd_lines)
    # elif num_of_per == 0:
    # angle, line_class = calculate_rad_per2(mlsd_lines, height)
    if perspective == 1:
        angle, line_class = calculate_rad_per1(mlsd_lines)
    elif perspective == 2:
        angle, line_class = calculate_rad_per2(mlsd_lines, height)

    if angle is not None and line_class is not None:
        df_lines = pl.DataFrame(
            {
                "class": line_class,
                "angle": angle,
                "X1": mlsd_lines[:, 0],
                "Y1": mlsd_lines[:, 1],
                "X2": mlsd_lines[:, 2],
                "Y2": mlsd_lines[:, 3],
            }
        )
        df_sphere_lines = pl.DataFrame(
            {
                "class": line_class,
                "angle": angle,
                "X1": mlsd_lines_sphere[:, 0],
                "Y1": mlsd_lines_sphere[:, 1],
                "X2": mlsd_lines_sphere[:, 2],
                "Y2": mlsd_lines_sphere[:, 3],
            }
        )
        return df_lines, df_sphere_lines, line_class
    else:
        raise ValueError("Unsupported number of perspectives: {}".format(num_of_per))


def change_dim_2Dto3D(line_segments):
    lines = np.zeros((line_segments.shape[0], 3))
    for i in range(line_segments.shape[0]):
        ls = line_segments[i, :]
        p1 = np.array([ls[0], ls[1], 1])
        p2 = np.array([ls[2], ls[3], 1])
        line = np.cross(p1, p2)
        lines[i, :] = line.copy()
    return lines


def project_to_sphere(points):
    # 点を半径1の球面に投影する
    points = np.atleast_2d(points)
    norm = np.linalg.norm(points, axis=1)
    return points / norm[:, np.newaxis]


def interpolate_sphere_line(p1, p2, num_points=10):
    # p1とp2間の球面上の点を補間する
    # 線分の両端点を球面に投影
    p1_sphere = project_to_sphere(p1).flatten()
    p2_sphere = project_to_sphere(p2).flatten()
    # 球面上の補間点を計算
    t = np.linspace(0, 1, num_points)[:, np.newaxis]  # tを列ベクトルに変換
    dot_product = np.dot(p1_sphere, p2_sphere)
    theta = np.arccos(dot_product)
    sphere_line = np.sin((1 - t) * theta) * p1_sphere + np.sin(t * theta) * p2_sphere
    # 各点を正規化
    sphere_line /= np.linalg.norm(sphere_line, axis=1)[:, np.newaxis]
    return sphere_line


def create_colored_lines(sphere_lines, class_list):
    # 線分の色を設定する
    colors = {
        0: [1, 0, 0],  # 赤
        1: [0, 1, 0],  # 緑
        2: [0, 0, 1],  # 青
    }
    line_set = o3d.geometry.LineSet()
    points = []
    lines = []
    line_colors = []

    for i, line in enumerate(sphere_lines):
        # 線分の両端点を取得
        p1 = np.array([line[0], line[1], 1])
        p2 = np.array([line[2], line[3], 1])
        # 球面上の補間点を計算
        sphere_arc = interpolate_sphere_line(p1, p2)
        start_idx = len(points)
        points.extend(sphere_arc)
        # 線分のリストにインデックスを追加
        for j in range(len(sphere_arc) - 1):
            lines.append([start_idx + j, start_idx + j + 1])
        # 色のリストに色を追加
        line_colors.extend([colors[class_list[i]]] * (len(sphere_arc) - 1))

    # 線分と色をLineSetに設定
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    return line_set

def project_to_gaussian_sphere():
    """2D点群をガウス球に投影する関数

    Args:
        image_path (str): 点群が含まれる画像のパス

    Returns:
        points_3d (np.ndarray): ガウス球上の3D点群
    """
    # 画像を読み込む
    image = cv2.imread("./PointCloud2D.png", cv2.IMREAD_UNCHANGED)
    height, width, _ = image.shape

    # 画像の幅と高さの最大値でスケーリングするための値
    scale = max(width, height)/1.6


    # 中心を原点とする座標系に変換するためのオフセット
    center_x, center_y = width / 2.0, height / 2.0

    # 3D点群を格納するリスト
    points_3d = []
    colors = []

    for y in range(height):
        for x in range(width):
            if image[y, x, 3] != 0:  # アルファチャンネルが0でない
                # 画像の中心を原点とする座標系に変換
                norm_x = (x - center_x) / scale
                norm_y = (y - center_y) / scale
                # ガウス球に投影（z座標の計算）
                z = np.sqrt(1 - norm_x**2 - norm_y**2) if norm_x**2 + norm_y**2 <= 1 else 0
                # y座標は画像とOpenGLの座標系の違いを考慮して反転
                points_3d.append([norm_x, -norm_y, z])
                # 色情報を追加（アルファチャンネルは無視）
                # OpenCVはBGR形式で色を読み込むため、RGB形式に変換
                colors.append(image[y, x, [2, 1, 0]] / 255.0)  # 正規化された色情報

    # Open3Dで点群を表示する
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

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

def get_point_cloud_data(modified_pcd,
                        sensor_size,
                        focal_length,
                        fov_vertical,
                        aspect_ratio,
                        near_clip,
                        far_clip,
                        camera_position,
                        camera_rotation,
                        image_width,
                        image_height):
    focal_length_x = (focal_length / sensor_size[0]) * image_width
    focal_length_y = (focal_length / sensor_size[1]) * image_height
    # カメラ行列を計算
    K = np.array([[focal_length_x, 0, image_width / 2],
                [0, focal_length_y, image_height / 2],
                [0, 0, 1]])

    # 点群データを読み込む
    points = np.asarray(modified_pcd.points)
    colors = np.asarray(modified_pcd.colors)
    normals = np.asarray(modified_pcd.normals)

    camera_rotation_radians = np.radians(camera_rotation)
    #print("camera_rotation_radians: ", camera_rotation_radians)


    # カメラの外部パラメータ行列を作成
    R = o3d.geometry.get_rotation_matrix_from_xyz(camera_rotation_radians)
    #print("R: ", R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = camera_position

    # 点群データをカメラ座標系に変換
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_camera = np.linalg.inv(T) @ points_homogeneous.T

    # カメラの内部パラメータ行列を使用して2Dに投影
    points_2d_homogeneous = (K @ points_camera[:3, :]).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
    points_2d[:, 0] = image_width - points_2d[:, 0]  # x軸の情報を反転
    points_2d[:, 1] = image_height - points_2d[:, 1]  # y軸の情報を反転

    # 2D座標と色情報を含む画像を作成
    data_list = []
    image = np.zeros((image_height, image_width, 4), dtype=np.uint8)  # アルファチャンネルを含む
    for i, (x, y) in enumerate(points_2d):
        if 0 <= x < image_width and 0 <= y < image_height:
            # 3D座標と法線ベクトル
            X_3d, Y_3d, Z_3d = points[i]
            nx, ny, nz = normals[i]

            # 色情報の取得（アルファ値を含む）
            color = (colors[i] * 255).astype(np.uint8)
            R, G, B = color  # RGBA値

            image[int(y), int(x), :3] = color[::-1]  # BGR形式
            image[int(y), int(x), 3] = 255  # アルファ値を最大に

            # 辞書型のデータをリストに追加
            data_list.append({
                "2d": [int(x), image_height - int(y)],
                "3d": [X_3d, Y_3d, Z_3d, nx, ny, nz],
                "color": [B, G, R, 255]
            })

    cv2.imwrite('PointCloud2D.png', image)

    return data_list

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
    # 回転行列を計算
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(camera_rotation_radians)

    # TODO: modify
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


# 各平面の中心点を計算する関数
def compute_plane_center(corners):
    return np.mean(np.array(corners), axis=0)

# 視錐台の中心点を計算する関数
def compute_frustum_center(near_corners, far_corners):
    near_center = np.mean(np.array(near_corners), axis=0)
    far_center = np.mean(np.array(far_corners), axis=0)
    return (near_center + far_center) / 2

# 点が平面の内側にあるかどうかを判定する関数
def is_point_inside_plane(point, plane_corners, plane_normal):
    plane_point = plane_corners[0]  # 平面上の任意の点
    point_vector = point - plane_point
    return np.dot(plane_normal, point_vector) <= 0

# 点がすべての平面の内側にあるかどうかを判定する関数
def is_point_inside_all_planes(point, plane_corners_normals):
    return all(is_point_inside_plane(point, corners, normal) for corners, normal in plane_corners_normals)

# 各平面の法線ベクトルを計算
def compute_plane_normal(corners):
    p1, p2, p3 = corners[:3]
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    return normal

def modify_point_cloud(ply_path,
                        sensor_size,
                        focal_length,
                        fov_vertical,
                        aspect_ratio,
                        near_clip,
                        far_clip,
                        camera_position,
                        camera_rotation,
                        image_width,
                        image_height):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    #print("pcd", pcd)

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
    frustum_mesh = create_frustum_mesh(sensor_size, fov_vertical, aspect_ratio, near_clip, far_clip, camera_position, camera_rotation)

    # LineSetのpoints属性から視錐台のコーナーを取得
    corners = np.asarray(frustum_mesh.points)

    # 各平面の頂点を定義
    near_plane_corners = corners[:4]
    far_plane_corners = corners[4:]
    left_plane_corners = [corners[i] for i in [1, 5, 6, 2]]
    right_plane_corners = [corners[i] for i in [0, 4, 7, 3]]
    top_plane_corners = [corners[i] for i in [0, 1, 4, 5]]
    bottom_plane_corners = [corners[i] for i in [2, 3, 6, 7]]

    # 点群のポイントを取得
    pcd_points = np.asarray(pcd.points)

    # # 点群のz軸が視錐台のnear平面より大きく、far平面より小さいものだけを残す
    # near_clip_z = near_plane_corners[0][2]  # near平面のz軸座標を取得
    # far_clip_z = far_plane_corners[0][2]    # far平面のz軸座標を取得

    # 点群のz軸が視錐台のnear平面より大きく、far平面より小さいものだけを残す
    combined_corners = np.concatenate((near_plane_corners, far_plane_corners), axis=0)
    near_clip_z = min(corner[2] for corner in combined_corners)
    far_clip_z = max(corner[2] for corner in combined_corners)

    # 条件に合う点だけをフィルタリング
    filtered_indices = np.where((pcd_points[:, 2] > near_clip_z) & (pcd_points[:, 2] < far_clip_z))[0]

    # 各平面の法線ベクトルを計算し、必要に応じて反転
    near_plane_normal = compute_plane_normal(near_plane_corners) * -1
    far_plane_normal = compute_plane_normal(far_plane_corners)
    left_plane_normal = compute_plane_normal(left_plane_corners) * -1
    right_plane_normal = compute_plane_normal(right_plane_corners)
    top_plane_normal = compute_plane_normal(top_plane_corners)
    bottom_plane_normal = compute_plane_normal(bottom_plane_corners)

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

    #o3d.visualization.draw_geometries([inside_all_planes_pcd,
    #                               frustum_mesh,])

    return inside_all_planes_pcd

def update_pcd(mlsd_lines, class_list, pcd_list, image_height):
    #print("len of pcd_list: ", len(pcd_list))
    # 線分の色を定義する
    colors = {
        0: (255, 0, 0),  # 赤
        1: (0, 255, 0),  # 緑
        2: (0, 0, 255),  # 青
    }
    # 線分のパラメータを事前に計算
    line_params = []
    for line in tqdm(mlsd_lines, desc="Calculating line parameters"):
        x1, y1, x2, y2 = line[:4]
        y1 = image_height - 1 - y1
        y2 = image_height - 1 - y2
        if x2 - x1 != 0:  # 垂直な線分を避ける
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            line_params.append((slope, intercept, True))
        else:
            line_params.append((None, x1, False))  # 垂直線の場合

    for point in pcd_list:
        point["3d"][2]=-point["3d"][2]
        point["3d"][0]=-point["3d"][0]

    # 線分に近い点のcolor_classを更新
    # for (line, class_id, (slope, intercept, is_not_vertical)) in tqdm(zip(mlsd_lines, class_list, line_params), desc="Updating PCD colors"):
    #     x1, y1, x2, y2 = line[:4]
    #     y1 = image_height - 1 - y1
    #     y2 = image_height - 1 - y2
    #     color = colors[int(class_id)]
    #     x_min, x_max = min(x1, x2), max(x1, x2)
    #     y_min, y_max = min(y1, y2), max(y1, y2)

    #     for pcd in pcd_list:
    #         x, y = pcd["2d"]
    #         if is_not_vertical:
    #             # 点(x, y)が線分の周り5px以内にあるかどうかをチェック
    #             distance = abs(slope * x - y + intercept) / (slope**2 + 1)**0.5
    #             if distance <= 2 and x_min <= x <= x_max and y_min <= y <= y_max:
    #                 pcd["color_class"] = color  # color_classを更新
    #         else:  # 垂直な線分の場合
    #             if abs(x - intercept) <= 2 and y_min <= y <= y_max:
    #                 pcd["color_class"] = color  # color_classを更新

    # color_classが未設定のpcd要素に対して、元のcolorを設定
    for pcd in tqdm(pcd_list, desc="Setting original colors"):
        if "color_class" not in pcd:
            pcd["color_class"] = pcd["color"][:3]  # RGBAからRGBへ変換

    return pcd_list


def get_border(lines, class_list, vanish_point, width, height):

    border={
        "right_wall":[],
        "left_wall":[],
        "ceiling":[],
        "floor":[],
        "back_wall":[]
    }

    intersections=classify_intersections(lines, class_list, vanish_point)

    """get_corner_point"""

    #消失点をもとに四分割
    categorized_point = {
    category: {"points": [], "class_label": []}
    for category in ["upper_left", "upper_right", "lower_right", "lower_left"]
      } 

    oblique_intersections = intersections["side_wall"] + intersections["ceiling_floor"]

    for i, intersection in enumerate(oblique_intersections):
         class_id = 0 if i < len(intersections["side_wall"]) else 1
         category = "upper_right" if intersection[0] > vanish_point[0] and intersection[1] > vanish_point[1] else \
               "lower_right" if intersection[0] > vanish_point[0] else \
               "upper_left" if intersection[1] > vanish_point[1] else "lower_left"

         categorized_point[category]["points"].append(intersection)
         categorized_point[category]["class_label"].append(class_id)

    #print(categorized_point)
         
    oblique_lines= []
         
    for i in range(len(lines)):
        if(class_list[i]==0):
            oblique_lines.append(lines[i])


    border, corners=find_corner_distance(intersections, vanish_point, oblique_lines, width, height)



    """
    #クタスタリングを用いたcoener pointsの取得
    print(corners)
    border=corners
    border["right_wall"] += [find_corner_line(border["right_wall"][1], oblique_lines),find_corner_line(border["right_wall"][0], oblique_lines)]
    border["left_wall"] += [find_corner_line(border["left_wall"][1], oblique_lines),find_corner_line(border["left_wall"][0], oblique_lines)]
    border["floor"] += [find_corner_line(border["floor"][1], oblique_lines),find_corner_line(border["floor"][0], oblique_lines)]
    border["ceiling"] += [find_corner_line(border["ceiling"][1], oblique_lines),find_corner_line(border["ceiling"][0], oblique_lines)]

    corners += [find_corner_distance(categorized_point[category]["points"], categorized_point[category]["class_label"], width*height, vanish_point) for category in ["upper_right", "lower_right", "lower_left", "upper_left"]]

    border["back_wall"]=corners
    border["right_wall"] += [corners[0], corners[1]]
    border["left_wall"] += [corners[2],corners[3]]
    border["ceiling"] += [corners[0] ,corners[3]]
    border["floor"] += [corners[1] ,corners[2]]


    corner_1=find_corner_line(corners[0],oblique_lines) #右上
    corner_2=find_corner_line(corners[1],oblique_lines) #右下
    corner_3=find_corner_line(corners[2],oblique_lines) #左下
    corner_4=find_corner_line(corners[3],oblique_lines) #左上
    border["right_wall"] +=[ corner_2 , corner_1]
    border["left_wall"] += [corner_3 , corner_4]
    border["ceiling"] += [corner_1 , corner_4]
    border["floor"] += [corner_2 , corner_3]

    """


    return border, corners, intersections


def find_corner_line(corner, lines):
    close_distance=float('inf')
    point=(0,0)
    for line in lines:
        if (close_distance>calc_distance(corner, (line[0], line[1]))):
             close_distance=calc_distance(corner, (line[0], line[1]))
             point=(line[2],line[3])
        if (close_distance>calc_distance(corner, (line[2], line[3]))):
            close_distance=calc_distance(corner, (line[2], line[3]))
            point=(line[0],line[1])
    return point 

def get_all_intersections(lines, class_list):
    intersections = {
        "side_wall": [],
        "back_wall": [],
        "ceiling_floor": []
    }

    # 線分をクラスによって分類
    lines_by_class = {0: [], 1: [], 2: []}
    for line, class_id in zip(lines, class_list):
        lines_by_class[int(class_id)].append(line)

    
    # クラス1と2の交点を計算
    for line1 in lines_by_class[1]:
        for line2 in lines_by_class[2]:
            if cross_jadge(line1, line2):
                intersection = find_intersection(line1, line2)
                intersections["back_wall"].append(intersection)
                
    for line1 in lines_by_class[1]:
        for line0 in lines_by_class[0]:
            if cross_jadge(line1, line0):
                intersection = find_intersection(line1, line0)
                intersections["ceiling_floor"].append(intersection)

    for line2 in lines_by_class[2]:
        for line0 in lines_by_class[0]:
            if cross_jadge(line2, line0):
                intersection = find_intersection(line2, line0)
                intersections["side_wall"].append(intersection)


    return intersections

def classify_intersections(lines, class_list, vanish_point):

    """異なるクラスの線分の交点を計算し、それらを天井・床、横の壁、奥の壁に分類する関数"""
    inf=float('inf')
    nan=float('nan')

    #消失点から最も近い交点を求める関数
    def find_closest_intersection(line_a, lines_b):
        distance_vanish=inf
        close_intersection=(nan, nan)
        for line_b in lines_b:
            if cross_jadge(line_a, line_b):
                 intersection = find_intersection(line_a, line_b)       
                 distance=calc_distance(vanish_point, intersection)
                 if(distance_vanish>distance):
                   close_intersection=intersection
                   distance_vanish=distance

        return close_intersection if close_intersection != (nan, nan) else None

    # 線分をクラスによって分類
    lines_by_class = {0: [], 1: [], 2: []}
    for line, class_id in zip(lines, class_list):
        lines_by_class[int(class_id)].append(line)

    # 交点を計算
    intersections = {
        "side_wall": [],
        "back_wall": [],
        "ceiling_floor": []
    }

    # クラス0と1の交点を計算 消失点から近いもののみを取り出す
    for line0 in lines_by_class[0]:
        intersection = find_closest_intersection(line0, lines_by_class[1])
        if intersection:
            intersections["ceiling_floor"].append(intersection)
    

    # クラス1と2の交点を計算
    for line1 in lines_by_class[1]:
        num1=0
        for line2 in lines_by_class[2]:
            if cross_jadge(line1, line2):
                intersection = find_intersection(line1, line2)
                intersections["back_wall"].append(intersection)
                num1+=1
        if(num1!=2):
            del intersections["back_wall"][-(num1)]

    # クラス2と0の交点を計算 消失点に近いもののみ
    for line0 in lines_by_class[0]:
        intersection = find_closest_intersection(line0, lines_by_class[2])
        if intersection:
            intersections["side_wall"].append(intersection)

    #print(intersections)

    
    return intersections

def devide(vanish_point, points):
    division_points={
        "RU":[],
        "LU":[],
        "LL":[],
        "RL":[]
    }

    for x, y in points:
        if x >= vanish_point[0] and y < vanish_point[1]:
            division_points["RU"].append((x, y))
        elif x < vanish_point[0] and y < vanish_point[1]:
            division_points["LU"].append((x, y))
        elif x < vanish_point[0] and y >= vanish_point[1]:
            division_points["LL"].append((x, y))
        elif x >= vanish_point[0] and y >= vanish_point[1]:
            division_points["RL"].append((x, y))

    return division_points

def find_nearest_point(reference_point, points):

    nan=float("nan")

    close_distance=float("inf") 
    close_point=(nan,nan)

    for point in points:
        distance=calc_distance(reference_point, point)
        if close_distance>distance:
            close_distance=distance
            close_point=point

    return close_point

def find_farthest_point(reference_point, points):
    nan=float("nan")

    farthest_distance=0
    farthest_point=(nan,nan)

    for point in points:
        distance=calc_distance(reference_point, point)
        if farthest_distance<distance:
            farthest_distance=distance
            farthest_point=point

    return farthest_point

def find_corner_distance(intersections,vanish_point, oblique_lines, width, height):

    nan=float("nan")
    corners = {
    'side_wall': {
        'RU': [],
        'RL': [],
        'LU': [],
        'LL': []
     },
    'back_wall': {
        'RU': [],
        'RL': [],
        'LU': [],
        'LL': []
     },
    'ceiling_floor': {
        'RU': [],
        'RL': [],
        'LU': [],
        'LL': []
     }
    }

    border={
        "right_wall":[],
        "left_wall":[],
        "ceiling":[],
        "floor":[],
        "back_wall":[]
    }

     #corner pointを通る奥行きの線の終点を得る
    for claster in (["side_wall", "back_wall", "ceiling_floor"]):
        devided_intersecrions=devide(vanish_point, intersections[claster])
        for part in (["RU","RL","LL", "LU"]):
            corners[claster][part]=find_farthest_point(vanish_point, devided_intersecrions[part])

    #クラスのcorner pointがない場合はほかのクラスの交点を格納
    wall_orientations = {"right_wall": ["RL", "RU"], "left_wall": ["LL", "LU"]}
    ceiling_floor_orientations = {"ceiling": ["RU", "LU"], "floor": ["RL", "LL"]}

    for orientation, corner_keys in wall_orientations.items():
        border[orientation] += [corners["ceiling_floor"][corner_keys[0]], corners["back_wall"][corner_keys[0]]] if math.isnan(corners["side_wall"][corner_keys[0]][0]) else [corners["side_wall"][corner_keys[0]]]
        border[orientation] += [corners["back_wall"][corner_keys[1]], corners["ceiling_floor"][corner_keys[1]]] if math.isnan(corners["side_wall"][corner_keys[1]][0]) else [corners["side_wall"][corner_keys[1]]]

    for orientation, corner_keys in ceiling_floor_orientations.items():
        border[orientation] += [corners["side_wall"][corner_keys[0]], corners["back_wall"][corner_keys[0]]] if math.isnan(corners["ceiling_floor"][corner_keys[0]][0]) else [corners["ceiling_floor"][corner_keys[0]]]
        border[orientation] += [corners["back_wall"][corner_keys[1]], corners["side_wall"][corner_keys[1]]] if math.isnan(corners["ceiling_floor"][corner_keys[1]][0]) else [corners["ceiling_floor"][corner_keys[1]]]

    # back_wall
    for part in ["RL", "LL", "LU", "RU"]:
         border["back_wall"].append(corners["back_wall"][part])

    #corner pointを通る奥行きの線の終点を得る
    for claster in (["right_wall", "left_wall", "ceiling", "floor"]):  
        border[claster] += [find_corner_line(border[claster][len(border[claster])-1],oblique_lines)]  

    #画像の四隅の座標を格納 
    if (border["left_wall"][-1][0])<border["left_wall"][-1][1]:
        border["ceiling"].append((0, 0))
    else:
        border["left_wall"].append((0, 0))


    if (width-border["right_wall"][-1][0])<border["right_wall"][-1][1]:
        border["ceiling"].append((width, 0))
    else:
        border["right_wall"].append((width, 0))

   #corner pointを通る奥行きの線の終点を得る
    for claster in (["right_wall", "left_wall", "ceiling", "floor"]):  
        border[claster] += [find_corner_line(border[claster][0], oblique_lines)]  

   
    #画像の四隅の座標を格納
    if (border["left_wall"][-1][0])<(height-border["left_wall"][-1][1]):
        border["floor"].insert(len(border["floor"])-1,(0, height))
    else:
        border["left_wall"].insert(len(border["left_wall"])-1, (0, height))
    
    if (width-border["right_wall"][-1][0])<(height-border["right_wall"][-1][1]):
        border["floor"].insert(len(border["floor"])-1,(width, height))
    else:
        border["right_wall"].insert(len(border["right_wall"])-1,(width, height))

    
    return border, corners

#使ってない
def find_corner_clustering(intersections,class_label, img_size,vanish_point):

    # DBSCANによるクラスタリング
    eps=img_size/9000
    min_samples=2

    x=[intersection[0] for intersection in intersections]
    y=[intersection[1] for intersection in intersections]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(np.vstack([x, y]).T)

    #クラスごとに配列をわける
    corner_candidates = [{} for _ in range(len(np.unique(clusters)))]

    for i in range(len(np.unique(clusters))):
        corner_candidates[i]={
        "points": [],
        "class_labels": []
         }

    for i in range(len(clusters)):
        if clusters[i]==-1:
            continue
        corner_candidates[clusters[i]]["points"].append(intersections[i])
        corner_candidates[clusters[i]]["class_labels"].append(class_label[i])

    corners=[]

    for point in corner_candidates:
        #print("point_class",point["class_labels"])
        if((1 in (point["class_labels"])) and (0 in (point["class_labels"]))):
            corners.append( (
             sum(x for x, y in point["points"]) / len(point["points"]),
             sum(y for x, y in point["points"]) / len(point["points"])
             ))

    if (len(corners)==0):
        #corner=(float("inf"), float("inf"))
        corner=(0,0)
    elif (len(corners)>1):
        for point in corners:
             distance = calc_distance(point, vanish_point)
             if distance > max_distance:
                 max_distance = distance
                 corner = point
    else:
        corner=corners[0]

    return corner


def extend_line(lines):
    extended_lines=lines.copy()
    extend_rate=1/15
    
    for i in range(int(len(lines))):
        direction_vector=(lines[i][2]-lines[i][0],lines[i][3]-lines[i][1])
        #if(lines)
        extended_lines[i][0]=(-extend_rate)*direction_vector[0]+lines[i][0]
        extended_lines[i][1]=(-extend_rate)*direction_vector[1]+lines[i][1]
        extended_lines[i][2]=(extend_rate)*direction_vector[0]+lines[i][2]
        extended_lines[i][3]=(extend_rate)*direction_vector[1]+lines[i][3]
    

    return extended_lines

def find_intersection(line1, line2):

    A=(line1[0],line1[1])
    B=(line1[2],line1[3])
    C=(line2[0],line2[1])
    D=(line2[2],line2[3])
    
    vector_AC=((C[0]-A[0]), (C[1]-A[1]))
    r=((D[1]-C[1])*vector_AC[0]-((D[0]-C[0])*vector_AC[1]))/((B[0]-A[0])*(D[1]-C[1])-(B[1]-A[1])*(D[0]-C[0]))

    distance=((B[0]-A[0])*r, (B[1]-A[1])*r)
    intersection=(A[0]+distance[0], A[1]+distance[1])

    return intersection
    
def cross_jadge(line1, line2):

    def max_min_jadge(a, b, c, d):
        min_AB, max_AB=min(a, b), max(a, b)
        min_CD, max_CD=min(c, d), max(c, d)
        if min_AB > max_CD or max_AB < min_CD:
            return False
        return True

    A=(line1[0],line1[1])
    B=(line1[2],line1[3])
    C=(line2[0],line2[1])
    D=(line2[2],line2[3])

    #座標の最大値、最小値での判定
    if not max_min_jadge(A[0], B[0], C[0], D[0]):
        return False
    
    if not max_min_jadge(A[1], B[1], C[1], D[1]):
        return False
    
    #式による判定
    tc1 = (A[0]-B[0]) * (C[1]-A[1]) + (A[1]-B[1]) * (A[0]-C[0])
    tc2 = (A[0]-B[0]) * (D[1]-A[1]) + (A[1]-B[1]) * (A[0]-D[0])
    tc3 = (C[0]-D[0]) * (A[1]-C[1]) + (C[1]-D[1]) * (C[0]-A[0])
    tc4 = (C[0]-D[0]) * (B[1]-C[1]) + (C[1]-D[1]) * (C[0]-B[0])

    return tc1 * tc2 <=0 and tc3 * tc4 <=0

def calc_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

def  apply_mask(rgb, points, width, height):
    int_points = [(int(x), int(y)) for x, y in points]

    mask=np.zeros(rgb.shape, dtype = np.uint8)
    pts = np.array([int_points], dtype=np.int32)

    cv2.fillPoly(mask, pts, (255, 255, 255))
    rgb_and = cv2.bitwise_and(rgb, mask)

    return mask, rgb_and


def delete_surface(pcd_list, surface_mask):
    deleted_pcd_list=[]

    for i in range(len(pcd_list)):
        if(pcd_list[i]["2d"][1]<len(surface_mask)):
             if (surface_mask[pcd_list[i]["2d"][1]][pcd_list[i]["2d"][0]][0]==0):
                 deleted_pcd_list.append(pcd_list[i])

    return deleted_pcd_list











        



    
    