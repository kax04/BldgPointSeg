import numpy as np
import polars as pl

import line_classification as LC


def create_line_list(mlsd_lines_sphere, mlsd_lines, height):
    """
    Create dataframe of line list
    Input: mlsd_lines
    Output: dataframe
    """
    # TODO: すべての透視図に対して、線分の角度を計算する
    angle_per1, line_class_per1 = LC.calculate_rad_per1(mlsd_lines)
    angle_per2, line_class_per2 = LC.calculate_rad_per2(mlsd_lines, height)

    # TODO: MLを用いて、線分のクラスを分類

    df_lines = pl.DataFrame(
        {
            "class": class_,
            "angle": angle,
            "X1": mlsd_lines[:, 0],
            "Y1": mlsd_lines[:, 1],
            "X2": mlsd_lines[:, 2],
            "Y2": mlsd_lines[:, 3],
        }
    )
    df_sphere_lines = pl.DataFrame(
        {
            "class": class_,
            "angle": angle,
            "X1": mlsd_lines_sphere[:, 0],
            "Y1": mlsd_lines_sphere[:, 1],
            "X2": mlsd_lines_sphere[:, 2],
            "Y2": mlsd_lines_sphere[:, 3],
        }
    )
    return df_lines, df_sphere_lines, class_


def class_list(lines, class_list):
    lines_class1 = []
    lines_class2 = []
    lines_class3 = []

    for line, class_ in zip(lines, class_list):
        if class_ == 0:
            lines_class1.append(line)
        elif class_ == 1:
            lines_class2.append(line)
        elif class_ == 2:
            lines_class3.append(line)

    return combine_line_list(
        lines, np.array(lines_class1), np.array(lines_class2), np.array(lines_class3)
    )


def combine_line_list(lines, lines_class1, lines_class2, lines_class3):
    new_lines = []
    new_lines.append(lines)
    new_lines.append(lines_class1)
    new_lines.append(lines_class2)
    new_lines.append(lines_class3)
    # new_array = np.vstack(new_lines)
    return np.array(new_lines, dtype=object)
