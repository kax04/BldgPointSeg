# 室内レイアウト推定を用いた、建物点群の面削除

本リポジトリでは、**室内レイアウト推定（壁・床・天井の領域分割）を活用し、建物の点群データから特定の面（壁・床・天井）を削除**するツールを提供します。

![Image](https://github.com/user-attachments/assets/a35b03ef-4c7b-4cc1-a214-00cb81a37c1d)

## インストール
このコードは、Windows11, CUDA10.2, cuDNN v8.9.6でテストされています。
```
# 仮想環境の再構築
conda env create -n BldgPointSeg -f BldgSeg.yaml
```

## 使用するモデル
本プロジェクトでは、線検出モデル ([M-LSD](https://github.com/navervision/mlsd)) の学習済モデルを使用します。


## 実行
### ファイル構成

指定した入力画像ディレクトリには、以下の三つのデータが必要です。

* 部屋の点群データ : XYZ座標を含む点群
* 部屋の二次元画像 : 点群を投影した画像
* カメラパラメータ : 画像取得時のカメラ設定

```
INPUT_DIR/
├── *.ply  # 部屋の点群データ
├── *.jpeg  # 部屋の二次元画像
└── *.txt  # カメラパラメータ
```

### 実行
```
python main.py -lm PATH_TO_LINE_MODEL -i INPUT_DIR -t 0
```

### オプション説明
- `-lm PATH_TO_LINE_MODEL` : **線検出モデルのパス**
- `-i INPUT_DIR` : **入力画像ディレクトリ**
- `-t` : **削除対象の指定 (0: 右壁, 1: 左壁, 2: 後壁, 3: 床, 4: 天井)**
