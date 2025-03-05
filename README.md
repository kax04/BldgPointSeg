# 室内レイアウト推定を用いた、建物点群の面削除

室内の二次元画像を壁・床・天井に分割する技術である、室内レイアウト推定を用いて、部屋の点群データを壁・床・天井に分割し、指定した室内平面を削除する。

![Image](https://github.com/user-attachments/assets/a35b03ef-4c7b-4cc1-a214-00cb81a37c1d)

## 準備
このコードは、Windows11, CUDA10.2, cuDNN v8.9.6でテストされている。
```
# 仮想環境の再構築
conda env create -n BldgPointSeg -f BldgSeg.yaml
```
入力データとして、部屋の点群, 部屋の二次元画像, 二次元画像を撮影した際のカメラパラメータが必要である。

## 実行
```
# 除去するターゲットの指定 -t 0:right_wall, 1:left_wall, 2:back_wall, 3:floor, 4:ceiling

python main.py -i ./image/depth -t 0
```
