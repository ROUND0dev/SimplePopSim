# 人口推移シミュレーション(+社会保障関係費の計算シミュレーション)

## これは何？
2023年時点のデータを基に、日本の人口がどのように推移していくのかを計算する簡易的なシミュレーションです。
ついでに社会保障関係の費用のシミュレーションも出来ます。

## 使い方

1. 総務省統計局(https://www.stat.go.jp/data/jinsui/2023np/index.html)から
「第1表　年齢（各歳）、男女別人口及び人口性比―総人口、日本人人口（2023年10月1日現在）（エクセル：22KB）」をダウンロードします。

2. ダウンロードしたエクセルファイルをsimulator.pyと同じディレクトリ(フォルダ)に置きます。別のディレクトリに置きたい場合はread_excelのパスを適宜変更してください。
'''
df = pd.read_excel('./05k2023-1.xlsx')
'''

3. requirements.txtから必要なライブラリをインストールします。

4. 実行する事で100年間の人口推移を表したグラフが表示されます。

### シミュレーションに関係する設定について

このシミュレーションでは以下の5つのパラメータを設定する必要があります。

+ NUM_OF_GEN:シミュレーション年数
+ BIRTH_RATE:出生率
+ PER_INSULANCE_PREMIUM:1人当たりの社会保険料負担額(万円)
+ PER_TAX_SSC:1人当たりの税額のうち社会保障関係費に使われる額(万円)
+ PER_SEN_SSC:高齢者1人当たりの社会保障給付額(万円)

シミュレーション年数を変更したい場合はNUM_OF_GENを変更してください。
出生率を変更したい場合はBIRTH_RATEを変更してください。

## 連絡先

X(Twitter):https://x.com/Fall_in_the_web

Mail:util.round0@gmail.com
