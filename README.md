# thread_title_generator3

Transformerとsentencepieceを利用してなんJスレッドタイトルっぽい文書を自動生成する。

[thread_title_generator2](https://github.com/ykicisk/thread_title_generator2)の続き。

## 生成例

※内容がここに記載する上で相応しくないスレッドタイトルは除いています。全文は[output_samples](./output_samples)を確認してください。

スレタイの後ろの数値は後述のFilterによるスコアです。

### 制限なし

```
「はん」を「さ」に変えて予測変換が一致する唯一の球団 0.866355836391449
ワイの彼女「ふにゃふにゃふにゃふにゃゅ」ワイ「ふゅうごゅ?」 0.8488953113555908
コミケのオリジナルグッズガチでクオリティが高そうな件について<kusa> 0.814984917640686
コンビニでいつも悩んでいるオヤジにブチ切れて 0.8129239082336426
dmmでゲームやってるとやってるニキおる? 0.8119136095046997
ワイ「今日はリリースさてそろそろでしょ」上司「ちょっと待って!それ面白いやつくわ!」 0.8110376596450806
【悲報】古参オタクに意見を「お前のスレ」と書き込まれたこともなく何故か開く 0.8091040849685669
ワイ草野球で143キロ計測 0.8077213764190674
禁書の七人に一人無能がおるよな 0.8069646954536438
【画像】「私はロボットではありません」と主張する男がsnsに広がり始める 0.801096498966217
```

### Prefix「三大」

```
三大購入特典で必ず発送された商品「発送」「6時」「前売り」 0.8469894528388977
三大モテない奴は難しい言葉「努力以上の努力が必要」「努力は必ず責め」「代表」 0.8417271375656128
三大怖さがよくわからない単語 「マクナル」「マクド」 0.8301395177841187
三大気になっててくる車『山登る』『砂降り』『雪の降る街』『雪』 0.8113009333610535
三大深夜12時にありがちな事「伸びてるレス」「実は内容が急に伸びていた」 0.8103548884391785
三大理解不能な「若干滑ったな」「それでも否定コメしたらええんや」あとひとつは? 0.8086734414100647
三大アフィ大喜利本 「 頭 く い な っ て の い」 0.8043513894081116
三大森野メジャーで獲らない球団「日ハム」「ヤクルト」あと1つは? 0.8020216822624207
三大彼女連れてきた車「砂漠」「まや」「ゲド戦」 0.7955830097198486
三大なんかあるミステリー小説『ダディ』『ゲームの面別』 0.7863622903823853
```

### Prefix「なぞなぞ」

```
【なぞなぞ】平常時川で橋の上に座る? 0.897125244140625
【なぞなぞ】鉄板志の鉄が1つだけ買えるポジションってなーんだ? 0.8656874895095825
【なぞなぞ】屁がこえることが出来ない奴と良い法律で同居できる問題 0.8614312410354614
【なぞなぞ】2chではすぐngにするとその程度の書き込みができる人ってなーんだ? 0.8599361181259155
【なぞなぞ】『校長部類』と「先生」しか居ない説 0.8505473136901855
【なぞなぞ】トロースでかい野菜ってなーんだ? 0.8492956757545471
【なぞなぞ】野菜がパンしか食えない食べ物ってなーんだ? 0.8488330841064453
【なぞなぞ】女の子が大好きな男をデートに誘うために行くものってなーんだ? 0.8411668539047241
【なぞなぞ】痒くもなく終える前に食べるものはなんでしょう? 0.8402188420295715
【なぞなぞ】「悪の趣味」を使って文章を作りなさい 0.8386388421058655
```

## 使用方法

### Dockerの起動・終了

```
# Docker起動 (JupyterLabも立ち上がる)
$ sudo docker-compose up -d

# docker終了
$ sudo docker-compose down
```

### ノートブックの実行

JupyterLabにアクセスして、`workspace`ディレクトリにあるノートブックを実行します

```
http://<ホスト名>:8889/lab?
```

## モデル詳細

[thread_title_generator2](https://github.com/ykicisk/thread_title_generator2)と同様に
スレタイっぽい文章の生成器とそれをスコア付けする評価器の２つで構成されます。

本リポジトリでは、Chenらの[Adding A Filter Based on The Discriminator to Improve Unconditional Text Generation](https://arxiv.org/pdf/2004.02135.pdf)に従って、Generator・Filterと呼びます。

Generatorでたくさんスレタイを生成し、Filterで高スコアのスレタイのみを出力します。

更に詳細: https://ykicisk.hatenablog.com/entry/2021/02/21/194400
