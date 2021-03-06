こちらより。

> https://github.com/pcyin/pytorch_nmt

> https://github.com/pcyin/pytorch_basic_nmt


**Pytorchで書かれたニューラル翻訳モデル。**

256次元隠れサイズのLSTMでは、IWSLT 2014 Germen-Englishデータセット（Ranzato et al。、2015）で14000語/秒の学習速度と26.9 BLEUスコアを達成します。



## ファイル構造

- `nmt.py`: main file
- `vocab.py`: パラレルコーパスから、`.bin`語彙ファイルを生成するスクリプト
- `util.py`: ヘルパー系
- `run_raml_exp.py|test_raml_models.py`: 様々な温度設定のRAMLモデルをテストするヘルパースクリプト（詳細については、[Norouzi et al。、2016]を参照してください）



## データセット

私たちは、（Ranzato et al。、2015）[script]で使われているIWSLT 2014ドイツ語 - 英語翻訳タスクの前処理バージョンを提供します。 データセットをダウンロードするには：

```shell
wget http://www.cs.cmu.edu/~pengchey/iwslt2014_ende.zip
unzip iwslt2014_ende.zip
```

スクリプトを実行すると、IWSLT 2014データセットを含む`data/` フォルダが抽出されます。 このデータセットには150Kのドイツ語 - 英語訓練文があります。` data /`フォルダには、データセットのパブリックリリースのコピーが含まれています。接尾辞`*.wmixerprep`のファイルは、Ranzato(2015)らによって、長い文章は分割され、稀な単語は`<UNK>`トークンによって置き換える前処理が施されています。 トレーニング/開発のために事前処理されたトレーニングファイルを使用することもできます（または独自の前処理戦略を思いつくこともできます）。しかし、テストのためには、テストファイルの元のバージョン、つまり`test.de-en(de | en)`を使用する必要があります。 



## 使い方

- 語彙ファイルの生成

```
python vocab.py
```

- シンプルな最尤法による実行

```
. scripts/run_mle.sh
```

- RAML用のサンプルファイル生成

```
. scripts/gen_samples.sh
```

- 報酬による最尤法 (Norouzi et al., 2016)

```
. scripts/run_raml.sh
```

- Reinforcement Learning (Coming soon)



各実行可能スクリプト（nmt.py、vocab.py）には、dotoptを使用して注釈が付けられます。 完全な使用方法については、ソースファイルを参照してください。



まず、次のコマンドを使用してトレーニングデータからボキャブラリファイルを抽出します。

```python
python vocab.py \
    --train-src=data/train.de-en.de.wmixerprep \
    --train-tgt=data/train.de-en.en.wmixerprep \
    data/vocab.json
```

これにより、ボキャブラリファイルdata / vocab.jsonが生成されます。 このスクリプトには、カットオフ周波数や生成されるボキャブラリのサイズを制御するオプションもあります。

トレーニングと評価を開始するには、単にdata / train.shを実行します。 訓練とデコードの後、公式の評価スクリプトであるmulti-bleu.perlを呼び出して、デコード結果のコーパスレベルのBLEUスコアをゴールドスタンダードと比較します。





## [Reward Augmented Maximum Likelihood for Neural Structured Prediction](https://arxiv.org/pdf/1609.00150.pdf)

論文での実験設定

| 項目名           | 内容                                                         |
| ---------------- | ------------------------------------------------------------ |
| データ           | 学習：WMT'14英語対フランス（36Mペア）, テスト：newstest-2014（3003ペア） |
| vocab            | ソース言語には200K語、ターゲット言語には80Kの語彙,           |
| 前処理           | 最大80トークンまでの訓練文のみ扱う, レア文字をUNKに置き換え（最初と最後の文字を考慮？） |
| モデル           | Attensionベース エンコーダ/デコーダRNN LSTM                  |
| layer, cell数    | 3層, 1024セル                                                |
| ビームサイズ     | 8                                                            |
| 温度係数τ        | 0.85 τ∈{0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0}             |
| RAMLサンプリング | ハミング距離（挿入, 削除せず距離を編集）                     |
| attention        | softmax attention, 1024ユニット（encoder最後, decoder最初の隠れ層） |
| ベースモデル     | D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. ICLR, 2015. |
| 学習方法         | 非同期SGD（12レプリカ, モメンタムなし）, サイズ128のミニバッチ |
| 学習率           | 最初0.5, 800Kステップ後に指数関数的に0.05に減少していく      |
| step             | 110〜130万ステップ                                           |
| 推論時           | 出力文中のUNKトークンを、最大注意活性化(largest attention activations)に従ってソースワードをコピーすることによって置き換える。M.-T. Luong, I. Sutskever, Q. V. Le, O. Vinyals, and W. Zaremba. Addressing the rare word problem in neural machine translation. ACL, 2015. |



















