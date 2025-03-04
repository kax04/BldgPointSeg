import copy

import numpy as np

"""
want to classify with 3 classes using angle and semantic segmentation
"""


def caluculate_rad_per3(lines, height):
    """
    Calculate the radian of the line detection
    with x-axis for line classification about perspective 1
    Input: lines (N, 4)
    Output: angle, class
    """

    # Extract out line coordinates
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    # Get the direction vector of the line detection
    # Also normalize
    dx = x1 - x2
    dy = y1 - y2
    norm_factor = np.sqrt(dx * dx + dy * dy)
    dx /= norm_factor  # cos
    dy /= norm_factor  # sin

    rad = np.arctan2(dy, dx)  # -pi ~ pi
    deg_pi = np.degrees(rad) + np.degrees(np.pi)  # 0 ~ 360
    class_deg = copy.deepcopy(deg_pi)  # 0 ~ 360

    # center of height
    center_h = height / 2

    # center of line height
    center_lh = (y1 + y2) / 2

    for i, deg, center_lh in enumerate(zip(class_deg, center_lh)):
        if center_lh < center_h:
            if 0 <= deg < 70:
                class_deg[i] = 0  # have vanishing point 1
            elif 290 < deg <= 360:
                class_deg[i] = 1  # have vanishing point 2
            elif ((70 <= deg <= 90)) | ((270 <= deg <= 290)):
                class_deg[i] = 2  # have vanishing point 3
        else:
            if 0 <= deg < 70:
                class_deg[i] = 1  # have vanishing point 2
            elif 290 < deg <= 360:
                class_deg[i] = 0  # have vanishing point 1
            elif ((70 <= deg <= 90)) | ((270 <= deg <= 290)):
                class_deg[i] = 2  # have vanishing point 3

    return deg_pi, class_deg


def calculate_rad_per2(lines, height):
    """
    Calculate the radian of the line detection
    with x-axis for line classification about perspective 1
    Input: lines (N, 4)
    Output: angle, class
    """

    # Extract out line coordinates
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    # Get the direction vector of the line detection
    # Also normalize
    dx = x1 - x2
    dy = y1 - y2
    norm_factor = np.sqrt(dx * dx + dy * dy)
    dx /= norm_factor  # cos
    dy /= norm_factor  # sin

    rad = np.arctan2(dy, dx)  # -pi ~ pi
    deg_pi = np.degrees(rad) + np.degrees(np.pi)  # 0 ~ 360
    class_deg = copy.deepcopy(deg_pi)  # 0 ~ 360

    class_id = np.array(class_deg, dtype=int)

    # center of line height を計算するためにイテラブルなオブジェクトを作成
    center_lh = (
        (y1 + y2) / 2
        if isinstance((y1 + y2) / 2, np.ndarray)
        else np.array([(y1 + y2) / 2])
    )

    # 水平に近い線分の中点のy座標を集めるためのリスト
    horizontal_line_midpoints = []

    # 水平に近い線分を判定するための閾値
    horizontal_threshold = np.tan(np.radians(4))  # 例えば4度の傾き以下を水平に近いとする

    # 水平に近い線分の中点のy座標を集める
    for dy, dx, y1, y2 in zip(dy, dx, y1, y2):
        if abs(dy / dx) < horizontal_threshold:
            horizontal_line_midpoints.append((y1 + y2) / 2)

    # 水平に近い線分の中点のy座標の平均を計算する
    if horizontal_line_midpoints:
        center_h = np.mean(horizontal_line_midpoints)
    else:
        center_h = height / 2  # 水平に近い線分がない場合は元の方法を使用

    for i, (deg, center_lh_value) in enumerate(zip(class_deg, center_lh)):
        if center_lh_value < center_h:
            if 0 <= deg < 84:
                class_id[i] = 0  # have vanishing point 1
            elif 276 < deg <= 360:
                class_id[i] = 1  # have vanishing point 2
            elif ((84 <= deg <= 90)) | ((270 <= deg <= 276)):
                class_id[i] = 2  # vertical line
            else:
                class_id[i]=4
        else:
            if 0 <= deg < 84:
                class_id[i] = 1  # have vanishing point 2
            elif 276 < deg <= 360:
                class_id[i] = 0  # have vanishing point 1
            elif ((84 <= deg <= 90)) | ((270 <= deg <= 276)):
                class_id[i] = 2  # vertical line
            else:
                class_id[i]=4

        # print(class_id[0])

    return deg_pi, class_id


# ３つのクラスに分類する
def calculate_rad_per1(lines):
    """
    Calculate the radian of the line detection
    with x-axis for line classification about perspective 1
    Input: lines (N, 4)
    Output: angle, class
    """

    # Extract out line coordinates
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]

    # Get the direction vector of the line detection
    # Also normalize
    dx = x1 - x2
    dy = y1 - y2
    norm_factor = np.sqrt(dx * dx + dy * dy)
    dx /= norm_factor  # cos
    dy /= norm_factor  # sin

    rad = np.arctan2(dy, dx)  # -pi ~ pi
    deg_pi = np.degrees(rad) + np.degrees(np.pi)  # 0 ~ 360
    class_deg = copy.deepcopy(deg_pi)  # 0 ~ 360
    deg_pi_small = copy.deepcopy(deg_pi)  # 0 ~ 360
    for i, deg in enumerate(class_deg):
        if (
            ((6 < deg < 84))
            | ((96 < deg < 174))
            | ((186 < deg < 264))
            | ((276 < deg < 354))
        ):
            class_deg[i] = 0  # have vanishing point
        elif ((0 <= deg <= 6)) | ((174 <= deg <= 186)) | ((354 <= deg <= 360)):
            class_deg[i] = 1  # holizontal line
        elif ((84 <= deg <= 96)) | ((264 <= deg <= 276)):
            class_deg[i] = 2  # vertical line

    for i, deg in enumerate(deg_pi_small):
        if 180 < deg <= 360:
            deg_pi_small[i] = deg - 180
        if 90 < deg_pi_small[i] <= 180:
            deg_pi_small[i] = deg_pi_small[i] - 90
    return deg_pi_small, class_deg
