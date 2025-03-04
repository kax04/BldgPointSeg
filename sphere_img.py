import os
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils_our import pred_lines
import intersection as INTER
from itertools import combinations

# CATEGORIES = ["per1", "per2", "per3"]
CATEGORIES = ["per1"]


def norm_nvec(nvec):
    # calculate the norm of the vector
    norm_nvec = np.linalg.norm(nvec, axis=1)
    # normalize the vector
    unit_nvec = nvec / norm_nvec[:, np.newaxis]  # 大きさを1にした単位法線ベクトル
    n = unit_nvec.shape[0]
    result = []
    for i in range(n):
        result.append(
            (unit_nvec[i, 1] / np.linalg.norm(unit_nvec[i, 1])) * unit_nvec[i, :]
        )
    unit_nvec = np.array(result)
    return unit_nvec


def plot_nvec_to_sphere(point):
    r = 1
    theta_1_0 = np.linspace(0, 2 * np.pi, 100)  # θ_1は[0,π/2]の値をとる
    theta_2_0 = np.linspace(0, 2 * np.pi, 100)  # θ_2は[0,π/2]の値をとる
    theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)  # ２次元配列に変換
    x = np.cos(theta_2) * np.sin(theta_1) * r  # xの極座標表示
    y = np.sin(theta_2) * np.sin(theta_1) * r  # yの極座標表示
    z = np.cos(theta_1) * r  # zの極座標表示
    # (x, y, z)
    x_line = [0, 1, 0, 0, 0, 0, 0]
    y_line = [0, 0, 0, 1, 0, 0, 0]
    z_line = [0, 0, 0, 0, 0, 1, 0]

    point = norm_nvec(point)

    fig = plt.figure(figsize=(6, 6))  # 描画領域を作成
    ax = fig.add_subplot(111, projection="3d")  # 3Dの軸を作成
    ax.plot_surface(x, y, z, alpha=0.05)  # 球を３次元空間に表示
    ax.plot(x_line, y_line, z_line, "-", color="black", ms=0.3)
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], c="black")
    ax.set_box_aspect((1, 1, 1))
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")


def pil2cv(image):
    """PIL型 -> OpenCV型"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def save_image(image, image_path):
    """
    Function to save images to a folder path
    If the folder does not exist, create it
    Also, create image names
    If there is an image with the same name in the folder, change the name you made
    Input: image, image path
    """

    folder_path = os.path.dirname(image_path)
    new_folder_name = os.path.basename(folder_path) + "_sphere"
    new_folder_path = os.path.join(os.path.dirname(folder_path), new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    image_name = os.path.basename(image_path)
    extension_name = os.path.splitext(image_name)[1]
    image_name = os.path.splitext(image_name)[0]
    image_name = image_name + "_sphere" + extension_name
    image_path = os.path.join(new_folder_path, image_name)
    cv2.imwrite(image_path, image)


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    buf = fig2data(fig)
    w, h, _ = buf.shape
    im = Image.frombytes("RGBA", (w, h), buf.tobytes())
    return im


def fig2imgarr(fig):
    im = fig2img(fig)
    imarr = np.asarray(im)
    imarr = np.delete(imarr, 3, 2)
    return imarr


def sphere_line_plot(lines, num, alpha=0.1, f=1.0, alternative=False):
    a = np.linspace(-math.pi / 2, math.pi / 2, num=10000)
    fig = plt.figure()
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.axis([-math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2])
    fig.add_axes(ax)
    ax.set_facecolor((0, 0, 0))
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.axis([-math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2])

    for i in range(lines.shape[0]):
        lines[i, :2] *= f
        b = INTER.curve_eq(a, i, lines, alternative=False)
        ax.plot(a, b, "-", c=[1, 1, 1, alpha])

    img = fig2imgarr(fig)
    imgray = np.mean(img, axis=2).astype(np.uint8)
    # plt.close()
    intersections = []
    if num == 0:
        plt.close()
    else:
        pass
        # for i, j in combinations(range(lines.shape[0]), 2):
        #     lines[i, :2] *= f
        #     if num == 1 or num == 3:
        #         init = 0
        #         intersection_point_X = INTER.find_intersection(
        #             [i, j], lines, init, alternative=False
        #         )
        #         intersection_point_Y = np.array(
        #             INTER.curve_eq(
        #                 intersection_point_X[0], i, lines, alternative=False
        #             )
        #         )
        #         intersection = np.append(intersection_point_X, intersection_point_Y)
        #         intersections.append(intersection)
        #     elif num == 2:
        #         init = [-math.pi / 2, math.pi / 2]
        #         intersection_point_X = INTER.find_intersection(
        #             [i, j], lines, init, alternative=False
        #         ).reshape(2, 1)
        #         intersection_point_Y1 = np.array(
        #             INTER.curve_eq(
        #                 intersection_point_X[0], i, lines, alternative=False
        #             )
        #         )
        #         intersection_point_Y2 = np.array(
        #             INTER.curve_eq(
        #                 intersection_point_X[1], i, lines, alternative=False
        #             )
        #         )
        #         intersection1 = np.append(
        #             intersection_point_X[0], intersection_point_Y1
        #         )
        #         intersection2 = np.append(
        #             intersection_point_X[1], intersection_point_Y2
        #         )
        #         intersections.extend([intersection1, intersection2])

        # intersections = np.asarray(intersections, dtype=float)
        # density_max_center, cluster_centers, cluster_num = INTER.find_density_max(
        #     intersections[:, 0], intersections[:, 1], eps=0.02, min_samples=12
        # )
        #print(intersections)
        # ax.plot(intersections[:, 0], intersections[:, 1], "o", c=[1, 0, 0, 1], markersize=2)
        # if density_max_center is not None:
        #     ax.plot(density_max_center[0], density_max_center[1], "o", c=[0, 1, 0, 1], markersize=8)
        # plt.show()
        plt.close()

    return imgray


def get_sphere_img(lines, i, alpha=0.1, f=1.0):
    sphere_img = sphere_line_plot(lines, i, alpha=alpha, f=f, alternative=False)
    return sphere_img


def change_dim_2Dto3D(line_segments):
    lines = np.zeros((line_segments.shape[0], 3))
    for i in range(line_segments.shape[0]):
        ls = line_segments[i, :]
        p1 = np.array([ls[0], ls[1], 1])
        p2 = np.array([ls[2], ls[3], 1])
        line = np.cross(p1, p2)
        lines[i, :] = line.copy()
    return lines


def show_img(
    lines,
    class_list,
    image_path,
    sphere_img,
    sphere_img_class1,
    sphere_img_class2,
    sphere_img_class3,
):
    ori_image = cv2.imread(image_path, 1)
    class1_image = ori_image.copy()
    class2_image = ori_image.copy()
    class3_image = ori_image.copy()
    for (x1, y1, x2, y2), class_ in zip(lines, class_list):
        cv2.line(ori_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        if class_ == 0:
            cv2.line(
                class1_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
            )
        elif class_ == 1:
            cv2.line(
                class2_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )
        elif class_ == 2:
            cv2.line(
                class3_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
            )

    fig = plt.figure(figsize=(8, 9))
    fig.subplots_adjust(hspace=0.4, wspace=0.05)
    ax1 = fig.add_subplot(4, 2, 1)
    ax2 = fig.add_subplot(4, 2, 2)
    ax3 = fig.add_subplot(4, 2, 3)
    ax4 = fig.add_subplot(4, 2, 4)
    ax5 = fig.add_subplot(4, 2, 5)
    ax6 = fig.add_subplot(4, 2, 6)
    ax7 = fig.add_subplot(4, 2, 7)
    ax8 = fig.add_subplot(4, 2, 8)

    ax1.set_title("Original Image with mlsd lines", fontsize=8)
    ax2.set_title("Sphere Image with mlsd lines", fontsize=8)
    ax3.set_title("Original Image with mlsd lines and class 1", fontsize=8)
    ax4.set_title("Sphere Image with mlsd lines and class 1", fontsize=8)
    ax5.set_title("Original Image with mlsd lines and class 2", fontsize=8)
    ax6.set_title("Sphere Image with mlsd lines and class 2", fontsize=8)
    ax7.set_title("Original Image with mlsd lines and class 3", fontsize=8)
    ax8.set_title("Sphere Image with mlsd lines and class 3", fontsize=8)
    ax1.imshow(pil2cv(ori_image))
    # cv2.imwrite("./image/depth/mlsd_line/ori_image.jpg", ori_image)
    ax2.imshow(sphere_img, cmap="gray")
    ax3.imshow(pil2cv(class1_image))
    # cv2.imwrite("./image/depth/class1_line/ori_image.jpg", class1_image)
    ax4.imshow(sphere_img_class1, cmap="gray")
    # cv2.imwrite("./image/depth/class1_sphere/ori_image.jpg", sphere_img_class1)
    ax5.imshow(pil2cv(class2_image))
    # cv2.imwrite("./image/depth/class2_line/ori_image.jpg", class2_image)
    ax6.imshow(sphere_img_class2, cmap="gray")
    # cv2.imwrite("./image/depth/class2_sphere/ori_image.jpg", sphere_img_class2)
    ax7.imshow(pil2cv(class3_image))
    # cv2.imwrite("./image/depth/class3_line/ori_image.jpg", class3_image)
    ax8.imshow(sphere_img_class3, cmap="gray")
    # cv2.imwrite("./image/depth/class3_sphere/ori_image.jpg", sphere_img_class3)
    plt.show()


def show_vp(vp, image_path):
    ori_image = cv2.imread(image_path, 1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(pil2cv(ori_image))
    ax.plot(vp[:, 0], vp[:, 1], "x", color="red")
    plt.show()


def get_lines(
    image_path, interpreter, input_details, output_details, num_of_img, num_of_per
):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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


def get_path(input_path):
    """
    Function to get the path of folders and images from a folder path
    Use only "per1", "per2", and "per3" folders
    argument parser: folder path
    returns: Image paths per folder
    """

    folder_input = os.listdir(input_path)
    new_folder_input = []
    for f in folder_input:
        # if f == CATEGORIES[0] or f == CATEGORIES[1] or f == CATEGORIES[2]:
        if f == CATEGORIES[0]:
            new_folder_input.append(f)
    paths = []
    for folder in new_folder_input:
        folder_path = os.path.join(input_path, folder)
        image_paths = []
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)
            image_paths.append(image_path)
        paths.append(image_paths)
    return paths
