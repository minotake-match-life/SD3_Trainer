# SD3.5
学習→train.py
推論→inference.py
モデル→mmditx.py
ユーティリティ→dit_embedder.py, other_impls.py, sd3_impls.py, sd3_infer.py

学習は、SD3の論文を参考に、Rectified FlowとLogit Normal Samplingで実装しています。
推論は、元のSD3のコードを少しいじって、簡単にオイラー法を実装しています。推論時のシフトはデフォルトの値にしています。
ユーティリティのコードは基本的に公式リポジトリのものですが、一部変更しています。

両方とも適当なデータローダーにしているので、自分のデータセットに合わせてください。
また、学習済みの重みを適切にダウンロードする必要があります。
環境構築には、pyproject.tomlを参考にしてください。uvを推奨です。

# SD1.5
学習→sd1_train.py
推論→sd1_inference.py

Diffusersから読み込んで学習推論ができるようなコードになっています。
同様に、適切なデータローダーと学習済みの重みを用意してください。

# 注意点
LoRAやControlNetにはデフォルトで対応していません。
スパコンを使う場合には、diffusersやtransformersの重みは事前にダウンロードしておいてください。
DistributedDataParallelの設定はスパコン上でOpenMPIを使うことを想定しています。
