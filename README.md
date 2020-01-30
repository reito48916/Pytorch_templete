# Pytorch_templete
簡単なPytorch機械学習のためのテンプレ。

## 動くかの確認
まずはこのプロジェクトが動作する事を確認する。

### 1.プロジェクトのクローン
```
$ git clone https://github.com/reito48916/Pytorch_templete.git
```
として今いるディレクトリにPytorch_templeteディレクトリが出来ればよい。

### 2.必要なpythonライブラリのインポート
ひな型の時点で、このプロジェクトはPytorch Numpyライブラリを使用する。それがインストールされているか確認するためには、Pythonを立ち上げて
```
>>> import numpy
>>> import torch
```
としてエラーが出ていないか見ればよい。ただしPythonやライブラリのバージョン違いなどについては関知していないので、importが出来るのにプロジェクトが動作しない場合は十中八九バージョン違いが原因である。

### 3.動作
```
$ mkdir learning_result
$ python learn.py
$ python test.py
```
として、特にエラーもなくlearning_resultディレクトリにパラメータとロスの記録が出力されテストロスもlearning_lossの終盤レベルに低く出力出来たらテスト成功である。これで君もディープラーナー（笑）。

## 構成
このひな形は4つのファイルと出力用の1つのディレクトリで出来ている（ディレクトリは動かすまで空っぽ）。説明する。

### 1.dataset.py
Pytorchではdatasetとdataloaderの仕組みを使うと簡単に深層学習の工夫が施せるらしい。そのために、教師あり学習のX、YをPytorchのTensor型からdataloaderが扱えるdataset型に変換するクラスを用意した。まぁ深く考えずXとYをこれの中のクラスのコンストラクタに渡せばよい。

### 2.model.py
超簡単なMLPとCNNをPytorchように記述した。Linear層は入出力層数のみで定義できる。2dConv層はめちゃくちゃ設定可能なパラメータがあるが、最低限入出力チャネルとカーネルサイズを書いておけばあとは一般的な数値にしてくれる。ひな型の例では引数左からin_channel、out_channel、kernel_sizeである。CNNにある追加メソッドは、Conv層から全結合層に移る際に一次元にならすためのメソッドである。

### 3.learn.py
かなめ。numpyでデータを作り、Pytorch用に変換、それをデータセットとしモデルも引っ張ってきて学習ループを回すプログラムになっている。回帰ではMSELossを、分類ではCrossEntropyLossを使っておけば良いと思うが、CrossEntropyLossはその内部にsoftmax層を持っているのでネットワークの方に書かないように注意。

### 4.test.py
learn.pyで更新した重みを読み込んでテストするプログラム。恐らくPytorch内にもっとスマートにロスや正答率をバシッと出してくれる仕組みはあるのだろうが、学習と違ってきっちりかっちり動作を追えないと困るので泥臭くforループで確認をした方がいいと思う。
