import argparse
import time

import numpy as np
import tensorflow as tf
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import math

import create_list as CL
import intersection as INTER
import sphere_img as SI
from functions import (
    get_lines,
    get_path,
    create_line_list,
    change_dim_2Dto3D,
    create_colored_lines,
    project_to_gaussian_sphere,
    load_camera_parameters,
    get_point_cloud_data,
    modify_point_cloud,
    update_pcd,
    classify_intersections,
    extend_line, 
    #find_intersections,
    get_border, 
    apply_mask,
    delete_surface,
    get_all_intersections
)

# Set up argument parser + options
parser = argparse.ArgumentParser(description="Main script for feature")
parser.add_argument(
    "-i", "--image-path", help="Path to the input folder", required=True
)
parser.add_argument(
    "-p", "--perspective", default="1" ,help="Perspective of the image", required=False, type=int
)
parser.add_argument(
    "-t", "--target", help="Target for delection 0right_wall, 1left_wall, 2back_wall,  3floor, 4ceiling", required=True, type=int
)
parser.add_argument(
    "-m",
    "--model_path",
    default="tflite_models/M-LSD_512_large_fp32.tflite",
    type=str,
    help="path to tflite model",
)
args = parser.parse_args()


def runFunction(input_path, interpreter, input_details, output_details, perspective, target):

    run_start=time.time()
    """関数を実行する場所

    Args:
        input_path : 画像フォルダが入っているパス
        interpreter : tfliteモデル
        input_details : tfliteモデルの入力
        output_details : tfliteモデルの出力
    """
    # 画像フォルダのパスのリストを取得
    image_paths = get_path(input_path, perspective)
    vps_our = []
    time_our = []
    for num_of_per, per_path in enumerate(image_paths):
        for num_of_img, img_path in enumerate(per_path):
            start = time.time()
            print("Processing image:", img_path)

            # 画像を分類するためのデータ作成
            mlsd_lines_sphere, mlsd_lines, width, height = get_lines(
                img_path,
                interpreter,
                input_details,
                output_details,
                num_of_img,
                num_of_per,
            )

            # print("image size ", width*height)
            print("width : ",width)
            print("height:",height)


            # num_of_perによって線分のクラス分類手法を変更
            df_lines, df_sphere_lines, class_list = create_line_list(
                mlsd_lines_sphere, mlsd_lines, height, num_of_per, perspective
            )
            #print(class_list)

            #print(mlsd_lines)




            # 画像を読み込む
            img_mlsd_lines = cv2.imread(img_path)
            img_extended_lines = cv2.imread(img_path)
            img_corner_vanish = cv2.imread(img_path)
            img_all_intersections = cv2.imread(img_path)
            img_intersections = cv2.imread(img_path)
            image_border =cv2.imread(img_path)
            img_vanish=cv2.imread(img_path)
            img_pb=cv2.imread(img_path)


            # 線分の色を定義する
            colors = {
                0: (0, 0, 255),  # 赤
                1: (0, 255, 0),  # 緑
                2: (255, 0, 0),  # 青
                4:(255,255,255)
            }
            

            # mlsd_linesに含まれる各線分に対して描画を行う
            for line, class_id in zip(mlsd_lines, class_list):
                x1, y1, x2, y2 = line
                color = colors[class_id]  # class_listに基づいて色を取得
                cv2.line(
                    img_mlsd_lines, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2
                )

            # cv2.imshow("Image with Lines_original", img_mlsd_lines)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            # f = open('mlsd_lines.txt', 'w')
            # for line in mlsd_lines:
            #     f.write("[")
            #     f.write(str(line[0]))
            #     for i in range(len(line)-1):
            #         f.write(","+ str(line[i+1]))
            #     f.write("]")
            #     f.write("\n")
            # f.close()

            # f = open('class_list.txt', 'w')
            # for x in class_list:
            #     f.write(str(x) + "\n")
            # f.close()



            for i in range(len(class_list) - 1, -1, -1):
                if class_list[i] == 4:
                    #print(i)
                    mlsd_lines = np.delete(mlsd_lines, i, 0)
                    class_list = np.delete(class_list, i)

           # print(mlsd_lines)

            extended_lines = extend_line(mlsd_lines)

            for line, class_id in zip(extended_lines, class_list):
                x1, y1, x2, y2 = line
                color = colors[class_id]  # class_listに基づいて色を取得
                cv2.line(
                    img_extended_lines, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2
                )

            for line, class_id in zip(extended_lines, class_list):
                x1, y1, x2, y2 = line
                color = colors[class_id]  # class_listに基づいて色を取得
                cv2.line(
                    img_pb, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2
                )

            #消失点の取得
            sphere_lines = change_dim_2Dto3D(mlsd_lines_sphere)
            sphere_lines_2 = CL.class_list(sphere_lines, class_list)
            points_3D, points_2D = INTER.find_all_intersections(
                 sphere_lines_2[1:], width, height, f=1.0
             )
            
            #print("vp",points_2D)
            
            vanish=points_2D[0][:]

            # 消失点とcorner points, を表示
            x,y=vanish

            # 拡張する画像のサイズ（例えば、元の画像の幅と高さの1.2倍にする）
            extended_width = int(width * 3)
            extended_height = int(height * 1.2)

            # 画像を1/2に縮小
            scale_factor = 1
            img_vanish_resized = cv2.resize(img_vanish, (int(width * scale_factor), int(height * scale_factor)))

            # 縮小した画像の新しいサイズを取得
            new_height, new_width, _ = img_vanish_resized.shape

            # 拡張する画像のサイズ（黒で塗りつぶす）
            extended_height = int(new_height * 1.5)  # 拡張後の高さ
            extended_width = int(new_width * 4)    # 拡張後の幅

            # 拡張した画像を作成（黒で塗りつぶす）
            extended_image = np.zeros((extended_height, extended_width, 3), dtype=np.uint8)

            # 縮小した元の画像を拡張した画像の中心に配置
            start_x = int((extended_width - new_width) / 2)
            start_y = int((extended_height - new_height) / 2)
            end_x = start_x + new_width
            end_y = start_y + new_height
            extended_image[start_y:end_y, start_x:end_x] = img_vanish_resized

            for vp in points_2D:
                #print("orint_vp")
                x, y = vp
                scaled_x = int(x * scale_factor)
                scaled_y = int(y * scale_factor)
                extended_point_x = int(scaled_x + (extended_width - new_width) / 2)
                extended_point_y = int(scaled_y + (extended_height - new_height) / 2)
                cv2.circle(extended_image, (extended_point_x, extended_point_y), 5, (0, 255, 0), -1)  # 点を描画

            # 拡張した画像を表示
            # cv2.imshow("Extended Image", extended_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.circle(img_vanish, (int(x), int(y)), radius=5, color=(255, 255, 0), thickness=-1)
            # cv2.imshow("Image with vanishing point", img_vanish)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 境界線、corner points, 交点の取得
            borders, corners ,intersections=get_border(extended_lines, class_list, vanish, width,height)


            colors={
                "side_wall":(255, 0, 255),
                "back_wall":(0, 255, 255),
                "ceiling_floor":(255, 255, 255)
            }
            

            for surface in ["side_wall" ,"back_wall", "ceiling_floor"]:
                color=colors[surface]
                for part in (["RU","RL","LL", "LU"]):
                    x,y =corners[surface][part]
                    if not(math.isnan(x)):
                         cv2.circle(img_corner_vanish, (int(x), int(y)), radius=5, color=color, thickness=-1)

            for surface in ["side_wall" ,"back_wall", "ceiling_floor"]:
                color=colors[surface]
                for part in (["RU","RL","LL", "LU"]):
                    x,y =corners[surface][part]
                    if not(math.isnan(x)):
                         cv2.circle(img_pb, (int(x), int(y)), radius=5, color=color, thickness=-1)

                     
            all_intersections=get_all_intersections(extended_lines, class_list)
                         
            # 全ての交点を表示
            for intersection in all_intersections["ceiling_floor"]:
                 x, y = intersection
                 cv2.circle(img_all_intersections, (int(x), int(y)), radius=5, color=(255, 255, 255), thickness=-1)

            for intersection in all_intersections["side_wall"]:
                 x, y = intersection
                 cv2.circle(img_all_intersections, (int(x), int(y)), radius=5, color=(255, 0, 255), thickness=-1)

            for intersection in all_intersections["back_wall"]:
                 x, y = intersection
                 cv2.circle(img_all_intersections, (int(x), int(y)), radius=5, color=(0, 255, 255), thickness=-1)


            #交点を表示
            for intersection in intersections["ceiling_floor"]:
                 x, y = intersection
                 cv2.circle(img_intersections, (int(x), int(y)), radius=5, color=(255, 255, 255), thickness=-1)

            for intersection in intersections["side_wall"]:
                 x, y = intersection
                 cv2.circle(img_intersections, (int(x), int(y)), radius=5, color=(255, 0, 255), thickness=-1)

            for intersection in intersections["back_wall"]:
                 x, y = intersection
                 cv2.circle(img_intersections, (int(x), int(y)), radius=5, color=(0, 255, 255), thickness=-1)
            
            
            #境界線を表示   
            border_colors={
                "right_wall": (255, 128, 0),
                "left_wall":(0, 0, 255),
                "back_wall":(0, 255, 0),
                "floor":(255, 255, 0),
                "ceiling":(255, 0, 255)

            }
    
            for surface in ["right_wall", "left_wall", "back_wall", "floor", "ceiling"]:
                color=border_colors[surface]
                for i in range(len(borders[surface])):
                    x1, y1=borders[surface][i]
                    if (i==len(borders[surface])-1):
                         x2, y2=borders[surface][0]
                    else:
                         x2, y2=borders[surface][i+1]
                    cv2.line(
                        image_border, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2
                        )



    
            # 描画された画像を表示
            # cv2.imshow("Image with Lines_original", img_mlsd_lines)
            # cv2.imshow("Image with Lines_extended", img_extended_lines)
            # cv2.imshow("Image with corners", img_corner_vanish)
            # cv2.imshow("Image with constrained intersections", img_intersections)
            # cv2.imshow("Image with all intersections", img_all_intersections)
            # cv2.imshow("Image with borders", image_border)
            # cv2.imshow("Image with vanishing point", img_vanish)
            # cv2.imshow("Image with points and lines", img_pb)
            
            
            #マスクを取得
            img = cv2.imread(img_path)
            rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            masks, rgb_ands = zip(*[apply_mask(rgb, borders[wall], width, height) for wall in ["right_wall", "left_wall", "back_wall", "floor", "ceiling"]])
            mask = list(masks)
            rgb_and = list(rgb_ands)

            #マスク画像を表示
            #fig, ax = plt.subplots(2, round(len(mask)))
            #ax = axes.ravel()
            # for i in range(len(mask)):
            #      ax[0,i].imshow(mask[i])
            #      ax[0,i].set_xticks([])
            #      ax[0,i].set_yticks([])

            #      ax[1,i].imshow(rgb_and[i])
            #      ax[1,i].set_xticks([])
            #      ax[1,i].set_yticks([])

            # plt.imshow(mask[4])  # 1枚目のマスクを表示
            # plt.xticks([])  # x軸の目盛りを非表示に設定
            # plt.yticks([])  # y軸の目盛りを非表示に設定
            # plt.show()
            # plt.show()

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()



            # 画像のパスから同じファイル名でplyファイルとtxtファイルを読み込む
            ply_path = img_path.replace(".png", ".ply")
            txt_path = img_path.replace(".png", ".txt")

            #print("ply file:", ply_path)

            # txtファイルからカメラパラメータを読み込む
            sensor_size, focal_length, fov, aspect_ratio, near_clip, far_clip, position, rotation = load_camera_parameters(txt_path)

            #print("sensor_size:", sensor_size)
            # 点群を編集
            modified_pcd = modify_point_cloud(ply_path,
                                              sensor_size,
                                              focal_length,
                                              fov,
                                              aspect_ratio,
                                              near_clip,
                                              far_clip,
                                              position,
                                              rotation,
                                              width,
                                              height)
            
            #print("modi",modified_pcd)

            #output_file = "output_file.ply"
            #o3d.io.write_point_cloud(output_file, modified_pcd)

            # 編集した点群を2次元へ射影、リストに格納
            pcd_list = get_point_cloud_data(modified_pcd,
                                            sensor_size,
                                            focal_length,
                                            fov,
                                            aspect_ratio,
                                            near_clip,
                                            far_clip,
                                            position,
                                            rotation,
                                            width,
                                            height)


            pcd_list = update_pcd(mlsd_lines, class_list, pcd_list, height)
            #print("pcd list1 : ",pcd_list)

            # targetの点群を削除
            pcd_list=delete_surface(pcd_list, masks[target])

            # pcd_listから3D座標とcolor_classを抽出
            points = np.array([item['3d'][:3] for item in pcd_list])  # 最初の3要素がX, Y, Z座標
            colors = np.array([item['color_class'] for item in pcd_list])  # color_class
            normals = np.array([item['3d'][3:] for item in pcd_list])  # 最後の3要素が法線ベクトル

            #print(pcd_list)

            # Open3DのPointCloudオブジェクトを作成
            pcd_visual = o3d.geometry.PointCloud()
            pcd_visual.points = o3d.utility.Vector3dVector(points)
            pcd_visual.points = o3d.utility.Vector3dVector(pcd_visual.points)
            pcd_visual.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 色を[0, 1]の範囲に正規化
            pcd_visual.normals = o3d.utility.Vector3dVector(normals)

            run_end=time.time()

            print("run time :", run_end-run_start)

            # 点群を視覚化
            o3d.visualization.draw_geometries([pcd_visual])

            # # 線分を3次元に変換
            sphere_lines = change_dim_2Dto3D(mlsd_lines_sphere)


            # # Open3D LineSetを作成
            # line_set = create_colored_lines(mlsd_lines_sphere, class_list)

            # # 球を作成（半径1の球）
            # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            # sphere.compute_vertex_normals()
            # sphere.paint_uniform_color([0.9, 0.9, 0.9])  # 球の色を薄いグレーに設定

            # # 半透明のための設定
            # sphere_line_set = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
            # sphere_line_set.paint_uniform_color([0.7, 0.7, 0.7])  # 球の線の色を薄いグレーに設定

            # # XYZ軸を作成
            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #     size=0.8, origin=[0, 0, 0]
            # )

            # # 点群を表示するために点群データを読み込む
            # pcd = project_to_gaussian_sphere()

            # # LineSetと一緒に球とXYZ軸を表示
            # o3d.visualization.draw_geometries(
            #     [line_set, sphere_line_set, coordinate_frame, pcd]
            # )

            # sphere_lines = CL.class_list(sphere_lines, class_list)
            # # 線分を分類するためのデータ作成はここまで
            # points_3D, points_2D = INTER.find_all_intersections(
            #     sphere_lines[1:], width, height, f=1.0
            # )
            # 画像を分類するためのデータ作成はここまで
            # sphere_imgs = [SI.get_sphere_img(sphere_lines[i], i) for i in range(4)]
            # SI.show_img(mlsd_lines, class_list, img_path, *sphere_imgs)
            # SI.show_vp(points_2D, img_path)
            # INTER.plot_sphere(points_3D)
            # SI.save_image(sphere_imgs[0], img_path)
            end = time.time()
            # vps_our.append(points_3D)
            time_our.append(end - start)
    # print("vps_our:\n", np.array(vps_our))
    # print("time_our:\n", time_our)


def main():
    input_path = args.image_path
    model_path = args.model_path
    target_path = args.target

    # Load tflite model
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    perspective = args.perspective

    runFunction(input_path, interpreter, input_details, output_details, perspective, target_path)


if __name__ == "__main__":
    main()
