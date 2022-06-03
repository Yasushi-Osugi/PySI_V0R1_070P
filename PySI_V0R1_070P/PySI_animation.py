import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import csv

plt.rcParams['font.family'] = 'Meiryo'



file_names_list = []
reward_list     = []

# ファイルの内容を一行ずつ読み込む
with open('.\data\csv_file_name_list', 'r') as f0:
    file_data = f0.readlines()
    for line in file_data:
        file_names_list.append(line)

print('.\data\file_names_list',file_names_list)



with open('.\data\csv_reward_list', 'r') as f1:
    file_data = f1.readlines()
    for line in file_data:
        reward_list.append(line)

print('.\data\reward_list',reward_list)


fig = plt.figure()
ims = []

plt.cla()



# *********************
# csv file数の余りでloop
# *********************

len_file_names_list = 0
len_file_names_list = len(file_names_list) #@220306 file_names_list長さ

print('len_file_names_list',len_file_names_list)

def update(i):
    plt.cla()


# *********************
# csv file数の余りでloop
# *********************
    i_mod = i % len_file_names_list


    csv_file_name = file_names_list[i_mod]
    reward_val    = reward_list[i_mod]


    # rstrip() 関数を使用して、Python の文字列から改行文字を削除する
    new_file_name = csv_file_name.rstrip()


    print('i_mod csv file数の余りでloop',i_mod)
    print('new_file_name',new_file_name)

    df = pd.read_csv(new_file_name, index_col=0)

    sA  = df['supply_accume']
    sI  = df['supply_I']
    sP  = df['supply_P']

    sy  = []

    dA  = df['demand_accume']
    dCO = df['demand_CO']
    dS  = df['demand_S']

    dy  = []


    print('i',i)

    print('df2',df)

    print('sA',sA)
    print('sI',sI)
    print('sP',sP)

    print('dA',dA)
    print('dCO',dCO)
    print('dS',dS)

    print('sA.values',sA.values)


    w = 0.25 # 棒グラフの幅
    ind = np.arange(len(df)) # x方向の描画位置を決定するための配列

    # supply side 供給量

    sy = sA.values + sI.values
    dy = dA.values + dCO.values

    print('sy.values',sy)
    print('dy.values',dy)



    # supply side 供給量
    plt.bar(ind, sA, width=w, color='lightgrey', label='supply:accume')
    plt.bar(ind, sI, width=w, bottom=sA.values, color='r', label='supply:I')
    plt.bar(ind, sP, width=w, bottom=sy, color='orange', label='supply:P')

    # demand side 需要量
    plt.bar(ind+w, dA, width=w, color='lightgrey', label='demand:accume')
    plt.bar(ind+w, dCO, width=w, bottom=dA.values, color='b',label='demand:CO')
    plt.bar(ind+w, dS, width=w, bottom=dy, color='c', label='demand:S')

    plt.subplots_adjust(bottom=0.2) # 下余白調整(default=0.1)
    

    plt.xticks(ind+w/2, df.index, rotation=90) # x軸目盛の描画位置が2本の棒の間にくるように調整
    plt.ylabel('lots')

    title_name = "PSI demand and supply REWARD = " + str(reward_val)

    plt.title( title_name )

    plt.legend( loc='upper left' )


    #plt.ylim(0, 100)
    #plt.ylim(0, 2000)
    #plt.ylim(0, 4000)
    plt.ylim(0, 7000)


# 10枚のプロットを 50ms ごとに表示するアニメーション
ani = animation.FuncAnimation(fig, update, interval = 50, frames=range(len_file_names_list) , repeat=False )

ani.save('PSI_Plan_animation.gif', writer='pillow')

plt.show()