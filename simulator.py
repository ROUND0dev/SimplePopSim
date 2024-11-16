# module import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
パラメータの設定値

NUM_OF_GEN:シミュレーション年数
BIRTH_RATE:出生率
PER_INSULANCE_PREMIUM:1人当たりの社会保険料負担額(万円)
PER_TAX_SSC:1人当たりの税額のうち社会保障関係費に使われる額(万円)
PER_SEN_SSC:高齢者1人当たりの社会保障給付額(万円)
'''
NUM_OF_GEN = 100
BIRTH_RATE = 1.1
PER_INSULANCE_PREMIUM = 20.4
PER_TAX_SSC = 13.6
PER_SEN_SSC = 230.4

'''
総務省統計局より公表されている
第1表 年齢（各歳）、男女別人口及び人口性比―総人口、日本人人口（2023年10月1日現在）
https://www.stat.go.jp/data/jinsui/2023np/index.html
のデータを利用

read_excelの引数にダウンロードしたエクセルファイルのパスを指定
'''
df = pd.read_excel('./05k2023-1.xlsx')

def birth(birth_rate, pop_mother):
  '''
  出生数の計算
  出生率birth_rateを35で割り、15歳から49歳までの女性の人口で掛ける事で、出生数を計算
  出生数に対して男女比1.05:1と仮定して男女別の出生数を計算
  '''
  birth_pop = birth_rate / 35 * pop_mother
  birth_pop_male = (1.05 / 2.05) * birth_pop
  birth_pop_female = birth_pop - birth_pop_male
  return birth_pop_male, birth_pop_female

def calc_ssc(pop):
  '''
  総人口popから65歳未満の人口pop_actを差し引きし、PER_COSTを乗じる事で社会保障費を計算
  '''
  pop_senior = pop[65:]
  ssc = PER_SEN_SSC * np.sum(pop_senior)
  return ssc

def calc_rev(pop):
  '''
  社会保障関係費の歳入を計算
  
  insulance_rev:社会保険料の歳入
  1人当たりの社会保険料を現役世代の人口で掛ける事で計算

  tax_rev:税金
  社会保障関係費に使われる1人当たりの税額に、成年人口を掛ける事で計算
  '''
  pop_insulance_rev = np.sum(pop[18:65])
  pop_tax_rev = np.sum(pop[18:])
  insulance_rev = PER_INSULANCE_PREMIUM * pop_insulance_rev
  tax_rev = PER_TAX_SSC * pop_tax_rev
  return insulance_rev + tax_rev

def step(params):
  '''
  1年間の人口増減シミュレーション
  pop_m:男性の年齢別人口(0~99歳)
  pop_f:女性の年齢別人口(0~99歳)
  rate_death_m:男性の年齢別死亡率
  rate_death_f:女性の年齢別死亡率
  birth_m:男性の出生数
  birth_f:女性の出生数
  pop_next_m:1年先の男性の年齢別人口
  pop_next_f:1年先の女性の年齢別人口
  '''
  pop_m = params['pop_male']
  pop_f = params['pop_female']
  rate_death_m = params['rate_death_male']
  rate_death_f = params['rate_death_female']
  birth_m = params['birth_male']
  birth_f = params['birth_female']

  death_m = pop_m * rate_death_m
  death_f = pop_f * rate_death_f

  pop_m -= death_m
  pop_f -= death_f

  pop_next_m = np.roll(pop_m, 1)
  pop_next_f = np.roll(pop_f, 1)

  pop_next_m[0] = birth_m
  pop_next_f[0] = birth_f
  
  return pop_next_m, pop_next_f

def create_rate_death(male_5s, female_5s):
  '''
  年齢層別(5年ごと)の死亡率を年齢別死亡率に変換
  '''
  
  rate_death_male = []
  rate_death_female = []
  for i in range(100):
    rate_death_male.append(male_5s[int(i / 5)])
    rate_death_female.append(female_5s[int(i / 5)])
    
  rate_death_male[-1] = male_5s[-1]
  rate_death_female[-1] = female_5s[-1]

  return rate_death_male, rate_death_female

def simulate_ssc(**kwargs):
  '''
  人口推移+社会保障費の推移をシミュレーション
  '''
  
  init_pop_male = kwargs.get('init_pop_male')
  init_pop_female = kwargs.get('init_pop_female')
  rate_death_male = kwargs.get('rate_death_male')
  rate_death_female = kwargs.get('rate_death_female')
  pop_list = kwargs.get('pop_list')
  pop_list_male = kwargs.get('pop_list_male')
  pop_list_female = kwargs.get('pop_list_female')
  ssc_list = kwargs.get('ssc_list')
  rev_list = kwargs.get('rev_list')
  birth_pop_list = kwargs.get('birth_pop_list')

  pop_m = [ float(e) for e in init_pop_male ]
  pop_f = [ float(e) for e in init_pop_female ]

  for i in range(NUM_OF_GEN):
    np_pop_m = np.array(pop_m)
    np_pop_f = np.array(pop_f)
    
    '''
    合計特殊出生率の定義に従って出生数を計算するため、15歳から49歳までの女性の人口を求めている
    '''
    pop_mother = np.sum(np_pop_f[15:50])
    birth_male, birth_female = birth(BIRTH_RATE, pop_mother)
    
    params = { 'pop_male': np_pop_m, 'pop_female': np_pop_f, 'rate_death_male': rate_death_male, 'rate_death_female': rate_death_female,
              'birth_male': birth_male, 'birth_female': birth_female}
    pop_m, pop_f = step(params)

    pop = pop_m + pop_f
    pop_list.append(list(pop))
    pop_list_male.append(list(pop_m))
    pop_list_female.append(list(pop_f))
    ssc = calc_ssc(pop)
    rev = calc_rev(pop)
    ssc_list.append(ssc)
    rev_list.append(rev)

    birth_pop_list.append(birth_male + birth_female)

def main():
  '''
  エクセルファイルから初期値を設定
  '''
  init_pop_male_l = df.iloc[12:62, 2].tolist()
  init_pop_male_u = df.iloc[12:62, 11].tolist()
  init_pop_male_l.extend(init_pop_male_u)

  init_pop_female_l = df.iloc[12:62, 3].tolist()
  init_pop_female_u = df.iloc[12:62, 12].tolist()
  init_pop_female_l.extend(init_pop_female_u)

  init_pop_male = init_pop_male_l
  init_pop_female = init_pop_female_l

  '''
  年齢層ごとの10万人あたりの死亡率
  厚生労働省 平成２０年 人口動態統計月報年計（概数）の概況
  表６－１ 年齢（５歳階級）別にみた死亡数・死亡率（人口１０万対）
  https://www.mhlw.go.jp/toukei/saikin/hw/jinkou/geppo/nengai08/kekka3.html
  より
  '''
  rate_death_male_5s = [0.000735, 0.000108, 0.000109, 0.000341, 0.000581, 0.000649, 0.000773, 0.001039, 0.001577, 0.002513, 0.004049, 0.006587, 0.009775, 0.014555, 0.023981, 0.041697, 0.070413, 0.118574, 0.196260, 0.283818, 1.0]
  rate_death_female_5s = [0.000664, 0.000086, 0.000064, 0.000189, 0.000277, 0.000325, 0.000423, 0.000587, 0.000847, 0.001310, 0.001986, 0.002905, 0.004014, 0.005947, 0.010479, 0.018820, 0.035870, 0.070219, 0.131947, 0.212797, 1.0]
  
  '''
  年齢層別の人口となっているので、年齢別の人口に変換する
  '''
  rate_death_male, rate_death_female = create_rate_death(rate_death_male_5s, rate_death_female_5s)

  ssc_list = []
  rev_list = []
  birth_pop_list = [ ]
  pop_list = [ init_pop_male + init_pop_female ]
  pop_list_male = [ init_pop_male ]
  pop_list_female = [ init_pop_female ]
  simulate_ssc(init_pop_male=init_pop_male, init_pop_female=init_pop_female, rate_death_male=rate_death_male
               ,rate_death_female=rate_death_female, pop_list=pop_list, pop_list_male=pop_list_male, pop_list_female=pop_list_female
               ,ssc_list=ssc_list, rev_list=rev_list, birth_pop_list=birth_pop_list)
  
  '''
  人口の推移をグラフで表示
  '''
  x = np.linspace(0, NUM_OF_GEN, NUM_OF_GEN)
  summarize_pop = [ sum(pop_list[i]) for i in range(NUM_OF_GEN) ]
  plt.xlabel("year")
  plt.ylabel("population")
  plt.plot(x, summarize_pop, label="population changes")
  plt.legend()
  plt.show()

  '''
  社会保障費の歳入と支出額の推移をグラフで表示
  '''
  plt.xlabel("year")
  plt.ylabel("amount")
  plt.plot(x, ssc_list, label="ssc")
  plt.plot(x, rev_list, label="rev")
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()