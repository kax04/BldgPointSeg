import numpy as np
import warnings
import math
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.cluster import DBSCAN
from itertools import combinations

# define the curve equation
def curve_eq(x, i, coefficient, alternative=False):
    # print("coefficient:", coefficient)
    if alternative:
        return np.arctan(
            -coefficient[i, 2]
            / (np.cos(x) * coefficient[i, 0] + np.sin(x) * coefficient[i, 1])
        )
    else:
        # Conditional branching to avoid "RuntimeWarning: divide by zero encountered in true_divide"
        if coefficient[i, 1] != 0:
            return np.arctan(
                (-coefficient[i, 0] * np.sin(x) - coefficient[i, 2] * np.cos(x))
                / coefficient[i, 1]
            )
        else:
            return np.arctan(
                -coefficient[i, 2]
                / (np.cos(x) * coefficient[i, 0] + np.sin(x) * coefficient[i, 1])
            )


# define the function to find the intersection point
def find_intersection(i, coefficient, init, alternative=False):
    # RuntimeWarningを無視する
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return fsolve(
        lambda x: curve_eq(x, i[0], coefficient, alternative=False)
        - curve_eq(x, i[1], coefficient, alternative=False),
        init,
    )


def find_density_max(x, y, eps, min_samples):
    # DBSCANによるクラスタリング
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(np.vstack([x, y]).T)
    # print("クラスタ数：", len(np.unique(clusters)) - 1)

    # 各クラスタの中心座標を計算
    cluster_centers = []
    for cluster in np.unique(clusters):
        if cluster == -1:
            continue
        cluster_points = np.vstack([x[clusters == cluster], y[clusters == cluster]]).T
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)

    # クラスタの中心座標のうち、最も密度の高い座標を探す
    density_max = 0
    density_max_center = None
    # print(cluster_centers)
    for center in cluster_centers:
        density = np.sum(
            np.linalg.norm(np.vstack([x, y]).T - center, axis=1) <= eps
        ) / len(x)
        if density > density_max:
            density_max = density
            density_max_center = center
    cluster_centers = np.array(cluster_centers)
    #print(cluster_centers)

    return density_max_center, cluster_centers, len(np.unique(clusters)) - 1


def angle_to_point(angle):
    alpha = angle[0]
    beta = angle[1]

    point = np.zeros(3)

    point[1] = np.sin(beta)
    point[0] = np.sin(alpha) * np.cos(beta)
    point[2] = np.cos(alpha) * np.cos(beta)
    #print("sin:", np.sign(point[2]))

    #point *= np.sign(point[2])

    return point


def to_pixel(vpts, h, w, focal_length=1.0):
    # print("inde",vpts, h, w)
    x = vpts[:, 0] / vpts[:, 2] * focal_length * max(h, w) / 2.0 + w // 2
    y = -vpts[:, 1] / vpts[:, 2] * focal_length * max(h, w) / 2.0 + h // 2
    points_2D = np.stack([x, y], axis=1)
    return points_2D


def find_all_intersections(lines, width, height, f):
    points_3D = []
    points_2D = []
    for k in range(lines.shape[0]):
        classified_lines = lines[k]
        intersections = []
        for i, j in combinations(range(classified_lines.shape[0]), 2):
            classified_lines[i, :2] *= f
            if k == 0 or k == 2:
                init = 0
                intersection_point_X = find_intersection(
                    [i, j], classified_lines, init, alternative=False
                )
                intersection_point_Y = np.array(
                    curve_eq(
                        intersection_point_X[0], i, classified_lines, alternative=False
                    )
                )
                intersection = np.append(intersection_point_X, intersection_point_Y)
                intersections.append(intersection)
            elif k == 1:
                init = [-math.pi / 2, math.pi / 2]
                intersection_point_X = find_intersection(
                    [i, j], classified_lines, init, alternative=False
                ).reshape(2, 1)
                intersection_point_Y1 = np.array(
                    curve_eq(
                        intersection_point_X[0], i, classified_lines, alternative=False
                    )
                )
                intersection_point_Y2 = np.array(
                    curve_eq(
                        intersection_point_X[1], i, classified_lines, alternative=False
                    )
                )
                intersection1 = np.append(
                    intersection_point_X[0], intersection_point_Y1
                )
                intersection2 = np.append(
                    intersection_point_X[1], intersection_point_Y2
                )
                intersections.extend([intersection1, intersection2])

        intersections = np.asarray(intersections, dtype=float)
        density_max_center, cluster_centers, cluster_num = find_density_max(
            intersections[:, 0], intersections[:, 1], eps=0.02, min_samples=12
        )

        if density_max_center is not None:
            # print("None")
            point_3D = angle_to_point(density_max_center)
            points_3D.append(point_3D)
        else:
            pass
            # print("density_max_center is None")

    # print("vp1",points_3D)
    points_3D = np.array(points_3D)
    if points_3D.shape[0] == 3:
        pass
        #print("points_3D:\n", points_3D)
    elif points_3D.shape[0] == 2:
        # 外積を計算
        #print("points_3D:\n", points_3D)
        v3 = np.cross(points_3D[0], points_3D[1]).reshape(1, 3)
        points_3D = np.append(points_3D, v3, axis=0)
        if abs(points_3D[1, 0]) < abs(points_3D[2, 0]):
            points_3D = points_3D[[0, 2, 1], :]
        #print("points_3D:\n", points_3D)
    elif points_3D.shape[0] == 1:
        # 直行するベクトルを計算して，外積を計算
        # print("points_3D:\n", points_3D)
        v2 = orth(points_3D).reshape(1, 3)
        points_3D = np.append(points_3D, v2, axis=0)
        v3 = np.cross(points_3D, v2).reshape(1, 3)
        points_3D = np.append(points_3D, v3, axis=0)
        if abs(points_3D[1, 0]) < abs(points_3D[2, 0]):
            points_3D = points_3D[[0, 2, 1], :]
        #print("points_3D:\n", points_3D)
    points_2D = to_pixel(points_3D, height, width)
    #print("points_2D:\n", points_2D)

    return points_3D, points_2D


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


def plot_sphere(point):
    r = 1
    theta_1_0 = np.linspace(0, 2 * np.pi, 100)  # θ_1は[0,π/2]の値をとる
    theta_2_0 = np.linspace(0, 2 * np.pi, 100)  # θ_2は[0,π/2]の値をとる
    theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)  # ２次元配列に変換
    x = np.cos(theta_2) * np.sin(theta_1) * r  # xの極座標表示
    y = np.sin(theta_2) * np.sin(theta_1) * r  # yの極座標表示
    z = np.cos(theta_1) * r  # zの極座標表示

    fig = plt.figure(figsize=(6, 6))  # 描画領域を作成
    ax = fig.add_subplot(111, projection="3d")  # 3Dの軸を作成
    ax.plot_surface(x, y, z, alpha=0.05)  # 球を３次元空間に表示
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], c="black")
    ax.set_box_aspect((1, 1, 1))
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
