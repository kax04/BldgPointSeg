'''
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import numpy as np
import cv2
import copy


def pred_lines(image, interpreter, input_details, output_details, num_of_img, num_of_per,input_shape=[512, 512], score_thr=0.2, dist_thr=10.0):
    image = cv2.imread(image, -1)
    h, w = image.shape[:2]
    size = (h + w)/2
    #print("resized_img shape:", image.shape)
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]
    resized_img = cv2.resize(image, (input_shape[0],input_shape[1]), interpolation=cv2.INTER_AREA)
    one = np.ones([input_shape[0], input_shape[1], 1])
    #print("resized_img shape:", resized_img.shape)
    #print("one shape:", one.shape)

    if len(resized_img.shape)!=3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
        resized_image = np.concatenate([resized_img, one], axis=-1)
    else:
        resized_image = np.concatenate([resized_img, one], axis=-1)

    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')

    # オーバーフローチャンネルを除外する
    image_data = batch_image[:, :, :, :3]

    # 新しいオーバーフローチャンネルを作成する（全ての要素が1）
    overflow_channel = np.ones((input_shape[0], input_shape[1], 1), dtype=np.float32)

    # オーバーフローチャンネルを次元を追加して4次元の配列にする
    overflow_channel = np.expand_dims(overflow_channel, axis=0)

    # オーバーフローチャンネルを画像データに連結する
    batch_image_processed = np.concatenate([image_data, overflow_channel], axis=-1)

    # Interpreterにテンソルを設定する
    interpreter.set_tensor(input_details[0]['index'], batch_image_processed)

    #interpreter.set_tensor(input_details[0]['index'], batch_image)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pts = interpreter.get_tensor(output_details[0]['index'])[0]
    pts_score = interpreter.get_tensor(output_details[1]['index'])[0]
    vmap = interpreter.get_tensor(output_details[2]['index'])[0]

    start = vmap[:,:,:2] #(x1, y1) 256x256x2
    end = vmap[:,:,2:] #(x2, y2) 256x256x2
    test = np.sum((start - end) ** 2, axis=-1)
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1)) # 256x256(maybe size of image) take abs

    segments_list = []
    name_list = []

    for center, score in zip(pts, pts_score):
        y, x = center # =pts
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
            img_name = "per{0}_image{1:04d}".format(num_of_per + 1, num_of_img)
            name_list.append(img_name)


    lines = 2 * np.array(segments_list) # 256 > 512
    lines[:,0] = lines[:,0] * w_ratio #x1
    lines[:,1] = lines[:,1] * h_ratio #y1
    lines[:,2] = lines[:,2] * w_ratio #x2
    lines[:,3] = lines[:,3] * h_ratio #y2

    #return lines, name_list
    return lines
