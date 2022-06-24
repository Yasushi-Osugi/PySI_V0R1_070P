# coding: utf-8
#
#
#            Copyright (C) 2020 Yasushi Ohsugi
#            Copyright (C) 2021 Yasushi Ohsugi
#            Copyright (C) 2022 Yasushi Ohsugi
#
#            license follows MIT license

import numpy as np
import matplotlib.pyplot as plt


# ******************************
# PySI related module
# ******************************
from PySILib.PySI_library_V0R1_070 import *

from PySILib.PySI_env_V0R1_070 import *

from PySILib.PySI_PlanLot_V0R1_070 import *


# ******************************
# profileとPSI計画データ読込み loading 
# class PlanSpaceとLotSpaceの初期設定
# ******************************

# ******************************
# node_file_nameは将来的に"scm_tree"で定義したnode_nameで順次読込む
# ******************************
#print('loading plan')
#
#node_name = "Wch00"
#
#i_PlanSpace,i_LotSpace = load_plan( node_name )


# ******************************
# profile読込み  
# ******************************
profile_name = "PySI_Profile_std.csv" #初期設定のプロファイル名を宣言

plan_prof = {} #辞書型を宣言

plan_prof = read_plan_prof_csv( profile_name )


# ******************************
# PSI_data_file_name定義
# ******************************

PSI_data_file_name = "PySI_data_std_IO.csv" # PSI data IOファイル名を宣言


# node_name はnode_to 将来的にSCM tree node指定
#    node_name = "WCHxx" #node_nameを指定

PSI_data = [] #retuen用にリスト型を宣言
PSI_data = read_PSI_data_csv( PSI_data_file_name )

#scm拡張用
#PSI_data = read_PSI_data_scmtree( PSI_data_file_name, node_name )

#print( 'read PSI_data',PSI_data )


# *******************************
# instanciate class PlanSpace 初期設定
# *******************************

# reading "PySI_Profile_std.csv"
# setting planning parameters
#
# Plan_engine = "ML" or "FS" , cost marameters, planning one and so on
# ML:Machine Learning  FS:Fixed Sequence/Normal PSI
#

i_PlanSpace = PlanSpace( plan_prof, PSI_data )


# ******************************
# instanciate class LotSpace 初期設定
# ******************************
i_LotSpace = LotSpace( 54 )


# ******************************
# instanciate class PlanEnv 初期設定
# ******************************

plan_env = PlanEnv()


# ******************************
# episode_no for Machine Learning
# ******************************

#episode_no = 10 ####必ず10回以上回す output maxが10回目以降のmaxを拾う

episode_no = 20

#episode_no = 50

#episode_no = 100


# ******************************
# ML initianise and Q_learning modules
# ******************************

state = 0      #stateは (x,y) = x + 54 * y の数値で座標を定義

prev_state  = 0 
work_state  = 0 

# ******************************
#
# stateは計画状態
# place_lot時に、計画状態と一対一のposition=(x,y)を生成
# position=(x,y)=(week_no_year,lot_step_no)をstateとして取り扱う
#
# 3月の第5週は、年間の第13週目、pos=(13,0)
# week_no_year   = 13
# lot_step_no    = 0     # ロットを1つ積み上げた数

# (x,y)座標は1つの数字に相互変換する
#   x=week_no_year  y=lot_step
#   num = x + 54 * y
#
# ******************************
# x軸 : 0<=x<=54週 week_no_year 年週: 年間を通した週
#
#       年週と別に、月内の週week_no=[W1,W2,W3,W4,W5]がある
#
#       Qlearningでの取り扱いは、0スタートのweek_pos=[0,1,2,3,4]とする
#
#       get_actionは、week_pos=[0,1,2,3,4]から一つの行動を選択すること
#
#
# ******************************
#
# actionはplace_lotすることで、get_actionした週にロットを積み上げる
# get_actionでx軸を選択したら、actionしてy軸を生成して、pos(x,y)が決まる
#
# ******************************
# y軸 : lot_step_no 選択したx軸週にロットを積み上げた数
#
# 内部の処理上はlist.append(a_lot)したリストの要素数len(list.append(a_lot))
#
# ******************************


prev_reward = 0 
prev_profit = 0 

profit_accum_prev_week = 0
profit_accum_curr_week = 0

Qtable = {}
Qtable[state] = np.repeat(0.0, 5)  # actions=0-4 week=1-5

lot_space_Y_LOG    = {}
lot_space_Y_LOG[0] = []


# ******************************
# Q_learning modules
# ******************************

def observe(next_action,i_PlanSpace,month_no, calendar_act_weeks):

    week_pos     = next_action
    week_no      = week_pos + 1
    week_no_year = month2year_week( month_no, week_no )

    calendar_inact_weeks = act_inact_convert( calendar_act_weeks )

    #### calendar_inact_weeks == i_PlanSpace.act_week_poss

    week4_month_list = [1,2,4,5,7,8,10,11]

# *****************************
# actionできない環境制約の判定 < LotSpaceの世界において >
# *****************************
# LotSpaceの世界では、off_week_listでplace_lotの可否を判定する。
# 判定action可能かどうか
# 環境制約からaction不可能であれば、位置を保持してnegative rewardを返す

# ******************************
# 制約を判定
# 1) 小の月の第5週目
# 2) 長期休暇週
# 3) ユーザー指定の稼働・非稼働週指定　船便の有無など
# ******************************

# action可能かどうか判定
# 環境制約からaction不可能であれば、状態位置を保持してnegative rewardをセット

# ******************************
# 1) 小の月の第5週目の判定
# ******************************
    if week_pos == 4 : #### next_action=week_pos=4 week_no=第5週の月の判定

        if month_no in week4_month_list:

        #@memo month2year_week( month_no, week_no=5 )の4週月

        # ******************************
        # update act_week_poss = seletable_action_list 
        # ******************************
            if week_pos in i_PlanSpace.act_week_poss:

                #act_week_possから年週week_no_year=月週week_pos=next_action外し
                i_PlanSpace.act_week_poss.remove( week_pos )

                monthly_episode_end_flag = False
                reward = -1000000
                #reward = -1

                return next_action, reward , monthly_episode_end_flag, i_PlanSpace.act_week_poss

            else:
            ### 小の月のカレンダー制約の判定後に、再度next_actionが入って来た

                monthly_episode_end_flag = False

                reward = -1000000
                #reward = -1

            # 既にremove済み
            ## act_week_possから年週week_no_year=月週week_pos=next_actionを外す
            #i_PlanSpace.act_week_poss.remove( week_pos )

                return next_action, reward , monthly_episode_end_flag, i_PlanSpace.act_week_poss


# ******************************
# 2) 長期休暇週の判定
# ******************************
    if week_no_year in i_PlanSpace.off_week_no_year_list:
    #week_pos     = next_action
    #week_no      = week_pos + 1
    #week_no_year = month2year_week( month_no, week_no )

        # ******************************
        # update act_week_poss = seletable_action_list 
        # ******************************
        if week_pos in i_PlanSpace.act_week_poss:

            # act_week_possから年週week_no_year=月週week_pos=next_actionを外す
            i_PlanSpace.act_week_poss.remove( week_pos )

            monthly_episode_end_flag = False
            reward = -1000000
            #reward = -1

            return next_action, reward , monthly_episode_end_flag, i_PlanSpace.act_week_poss

        else:
        ### 長期休暇のカレンダー制約の判定後に、再度next_actionが入って来た

            monthly_episode_end_flag = False

            reward = -1000000
            #reward = -1

            # 既にremove済み
            ## act_week_possから年週week_no_year=月週week_pos=next_actionを外す
            #i_PlanSpace.act_week_poss.remove( week_pos )

            return next_action, reward , monthly_episode_end_flag, i_PlanSpace.act_week_poss


# ******************************
# 3) ユーザー指定の稼働・非稼働週指定　船便の有無など
# ******************************
    elif week_pos in calendar_inact_weeks:

    #### MEMO
    #week_pos     = next_action
    #week_no      = week_pos + 1
    #week_no_year = month2year_week( month_no, week_no )

        # ******************************
        # update act_week_poss = seletable_action_list 
        # ******************************
        if week_pos in i_PlanSpace.act_week_poss:

            # act_week_possから年週week_no_year=月週week_pos=next_actionを外す
            i_PlanSpace.act_week_poss.remove( week_pos )

            monthly_episode_end_flag = False
            reward = -1000000
            #reward = -1

            return next_action, reward , monthly_episode_end_flag, i_PlanSpace.act_week_poss

        else:
            ### 物流カレンダー制約の判定後に、再度next_actionが入って来た

            monthly_episode_end_flag = False

            reward = -1000000
            #reward = -1


            # 既にremove済み
            ## act_week_possから年週week_no_year=月週week_pos=next_actionを外す
            #i_PlanSpace.act_week_poss.remove( week_pos )

            return next_action, reward , monthly_episode_end_flag, i_PlanSpace.act_week_poss


# ******************************
# ACTION(=place_lot)  UPDATE(=calc_plan)  EVALUATION(=eval_plan)
# ******************************

    else:

# ******************************
# 新規ロット番号の付番
# ******************************
        i_PlanSpace.lot_no += 1  #@ 新規ロット番号の付番


# ******************************
# 実行actin place_lot / 状態の更新update_state calc_plan_PSI / 評価eval_plan
# ******************************
        next_state, reward, monthly_episode_end_flag = plan_env.act_state_eval(next_action, month_no, i_PlanSpace, i_LotSpace, episode)

        return next_state, reward, monthly_episode_end_flag, i_PlanSpace.act_week_poss


# ******************************
# get_action 5つの週から選択する week_pos=[0,1,2,3,4] +1 week_no=[1,2,3,4,5]
# ******************************
def get_action(state, Qtable, act_week_poss, episode):

    # e-greedy
    epsilon  = 0.2 

    #epsilon  = 0.5
    #epsilon  = 0.5 * (0.99 ** episode)  ### cartpoleのepsilon例


# ******************************
# plan_eigine="ML"を実行 確率epsilonでargmaxする。
# ******************************
    if i_PlanSpace.plan_engine == "ML":

        if  epsilon <= np.random.uniform(0, 1):
            ### exploit ###
            next_action = np.argmax(Qtable[state])

        else:  
            ### explore ###
            next_action = np.random.choice(act_week_poss) 

            ### 前処理の制約確認で選択可能な行動がact_week_possに入っている

            ### next_action = np.random.choice([0, 1, 2, 3, 4])
            ### 制約がなければ、5つの週を選択できる行動がact_week_possに入る


# ******************************
# plan_eigine="FS"を実行
# ******************************
    elif i_PlanSpace.plan_engine == "FS":

### ロットシーケンスlot_noを月内のaction可能数で割った余りで固定シーケンス発生
#
#w_mod = i_PlanSpace.lot_no % len(active_week) 
#
#next_action = active_week[w_mod]

        # ******************************
        # [0,1,2,3,4]の中にconstraint週が[1,4]ならselectableは[0,2,3]
        # ******************************
        next_action_list    = []
        next_action_list    = Qtable[state]

        ### act_week_poss # list of next_action
        selectable_pos_list = i_PlanSpace.act_week_poss 

        w_mod = i_PlanSpace.lot_no % len(selectable_pos_list) 

        next_action = selectable_pos_list[w_mod]

    else:
# ******************************
# plan_eigineを追加する場合はココでelifする
# ******************************
        print('No plan_engine definition')

    return next_action


# ******************************
# Qtableの更新
# ******************************
def update_Qtable(state, Qtable, next_state, reward, next_action):

# ******************************
# 新しい状態を判定してQtableに新規追加
# ******************************
    if next_state not in Qtable: #Q_tableにない状態を追加セット

        Qtable[next_state] = np.repeat(0.0, 5) ### action 0-4 = w1-w5

    cur_state = state

    state = next_state

# ******************************
# Q_table update
# ******************************
    alpha = 0.1   
    #alpha = 0.2  
    #alpha = 0.5  

    gamma = 0.99

    Qtable[cur_state][next_action]=(1-alpha)*Qtable[cur_state][next_action]+\
              alpha * (reward + gamma * max(Qtable[next_state]))

    return state, Qtable


# ******************************
# main process
# ******************************
if __name__ == '__main__':


    plan_reward = []    # reward logging

    monthly_episode_end_flag = False  # エージェントがゴールしてるかどうか？


    for episode in range(episode_no):

        print('START episode = ',episode_no)

        episode_reward = []   #  報酬log


# ******************************
# 次のepisodeスタート前にPSI_dataとlot_countsを0クリアする
# ******************************

        # 需要Sの入力データを保持する。需要はゼロクリアしない
        # i_PlanSpace.S_year

        i_PlanSpace.CO_year    = [ 0 for i in range(54)]
        i_PlanSpace.I_year     = [ 0 for i in range(54)]
        i_PlanSpace.P_year     = [ 0 for i in range(54)]
        i_PlanSpace.IP_year    = [ 0 for i in range(54)]

        i_PlanSpace.lot_counts = [ 0 for i in range(54)]


# ******************************
# Q学習は月次で12回の処理を実施する
# ******************************
        for month_no in range(1,13): #LOOP 12ヶ月分

            print('episode_no and month_no = ', episode_no, month_no )

            WEEK_NO = 5

            i_LotSpace.init_lot_space_M( WEEK_NO )
            #i_LotSpace.init_lot_space_M( 5 )


# ******************************
#   月のS==0で、i_PlanSpace.S445_month[month_no] == 0の時、月の処理をスキップ
# ******************************

            if i_PlanSpace.S445_month[month_no] == 0 :

                continue

# lot_place処理のためのget_actionをend_flagまで繰り返す。
# もし、月内にactive_weekがなければ、end_flag=Trueを返す。

            #辞書型で、{month_no,[week_list,,,]}のデータを持たせる
            calendar_week_dic = {} # active_week_dicの辞書型の宣言

            calendar_week_list = i_PlanSpace.calendar_cycle_week_list

#calendar_week_list = [1,3,5,7,9,11,14,16,17,20,22,24,27,29,31,33,35,37,40,42,44,46,48,50]

            calendar_week_dic = make_active_week_dic(calendar_week_list)

#an image
#calendar_week_dic {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4, 5], 4: [1, 2, 3, 4], 5: [1, 2, 3, 4], 6: [1, 2, 3, 4, 5], 7: [1, 2, 3, 4], 8: [1, 2, 3, 4], 9: [1, 2, 3, 4, 5], 10: [1, 2, 3, 4], 11: [1, 2, 3, 4], 12: [1, 2, 3, 4, 5]}

            calendar_act_week = []
            calendar_act_week = calendar_week_dic[month_no]

# カレンダー制約、1)小の月、2)長期休暇off週制約、3)船便物流制約を見て、
# active_weekSを生成する

# 制約共通のactive_weekS=[0,1,2,3,4]を定義する。
# 月内のplanningのconstraint checkでactive_weekS=[]となったらend_flag == True

            act_week_poss             = [0,1,2,3,4] ###初期化 local
            i_PlanSpace.act_week_poss = [0,1,2,3,4] ###初期化 i_PlanSpaceの中


            while not(monthly_episode_end_flag == True):    

            # 終了判定の基本は、 accume_P >= accume_Sで終了を判定する
            #
            # 変化形として、
            # 1) 安全在庫日数 safty_stock_daysを上乗せして終了判定するケース
            #
            #    accume_P >= accume_S + SS_days
            #
            # 2) accume_Profitが減少し始めた時に終了判定するケース
            #
            #    accume_Profit_current < accume_Profit_previous

                # ******************************
                # get_action
                # ******************************
                next_action = get_action(state, Qtable, act_week_poss, episode) 
# memo
# 機能拡張の案 get_actionには事前にpre_observeして評価する案が考えられる
# 例えば、
# 1. place_back_lotして状態を戻す。
# 2. PlanSpaceのcopy退避で状態を戻す。
# 3. check_action_constraint後、lot_countsを仮更新してEvalPalnSIPする

                # ******************************
                # monitor before_observe
                # ******************************
                pac = sum(i_PlanSpace.Profit[1:])
                #profit_accum_curr = sum(i_PlanSpace.Profit[1:])


                # ******************************
                # observe    check_action_constraint/action/update_state/eval
                # ******************************
                next_state, reward, monthly_episode_end_flag, act_week_poss = observe(next_action,i_PlanSpace, month_no, calendar_act_week) 


                # ******************************
                # monitor after_observe
                # ******************************
                pap = sum(i_PlanSpace.Profit[1:])
                #profit_accum_prev = sum(i_PlanSpace.Profit[1:])


                # ******************************
                # "PROFIT"の終了判定
                # ******************************
                if i_PlanSpace.reward_sw == "PROFIT":

                    if month_no >= 4: # 立ち上がりの3か月はそのまま

                        profit_deviation = ( pac - pap ) / pap

                        #print('profit_deviation',profit_deviation)


                # ******************************
                # 利益累計の変化を見てend conditionを設定する点に注目
                # ******************************
                        if profit_deviation <= 0:  #利益の変化率が0%以下

                        #if profit_deviation <= - 0.01:  #利益の変化率が-1%以下
                        #if (pac - pap) / pap <= - 0.03: #利益の変化率が-3%以下

                            monthly_episode_end_flag = True


                # ******************************
                # Q学習の処理　stateとrewardの変化からQ-Tableを管理
                # ******************************
                state, Qtable = update_Qtable(state,Qtable,next_state,reward,next_action) 

                episode_reward.append(reward)

                # M month
                #show_lot_space_M(i_LotSpace.lot_space_M)

                # Y year
                #show_lot_space_Y(i_LotSpace.lot_space_Y)

                # 制約loopから抜ける
                # ******************************
                # Q学習後、すべての制約を通過した結果act_week_poss== []なら終了
                # ******************************
                if act_week_poss == []: # 選択できるactive week positionsがない

                    monthly_episode_end_flag = true

            # ******************************
            # 月次終了のこの位置で、月次の操作域lot_space_Mを初期化
            # ******************************

            i_LotSpace.lot_space_M = [[] for j in range(5)] 

            monthly_episode_end_flag = False

            i_PlanSpace.lot_no = 0  ### クラス中の変数とする


        # ******************************
        # 年次終了の前に show_lot_space_Yを見たい時に使用
        # ******************************
        # print('episode NO',episode)
        # show_lot_space_Y(i_LotSpace.lot_space_Y)


        # ******************************
        # 年次終了の前に episode_noとshow_lot_space_YのLOG
        # ******************************
        lot_space_Y_LOG[episode] = i_LotSpace.lot_space_Y


# ******************************
# 次のepisodeスタート前にlot_space_Yを0クリアする
# ******************************
        i_LotSpace.lot_space_Y = [[] for j in range(53)]

        #print('lot_space_Yを0クリア')
        #show_lot_space_Y(i_LotSpace.lot_space_Y)


        # ******************************
        # episode reward log
        # ******************************
        plan_reward.append(np.sum(episode_reward))


        # ******************************
        # 年次終了のこの位置で、Q学習の初期化
        # ******************************
        state = plan_env.reset(i_LotSpace)           # init state

        state = 0             #stateは (x,y) = x + 54 * y の数値で座標を定義

        update_Qtable(state,Qtable,next_state,reward,next_action) 

        monthly_episode_end_flag = False


# ******************************
# pickup TOP reward plan    lot_space_Y[top_reward]
# ******************************

# ******************************
# episode毎にloggingした計画結果から、episode10回目以降でreward maxを取り出す
# ******************************

    max_value = max(plan_reward[9:]) ### episode10回目以降

    #print('plan_reward',plan_reward)
    #print('plan_reward[9:]',plan_reward[9:])
    #print('max_value',max_value)

    max_index = plan_reward.index(max_value)

    #print('max value and index',max_value,max_index)

    fin_lot_space_Y = lot_space_Y_LOG[max_index]

    #show_lot_space_Y(fin_lot_space_Y)


# ******************************
# episode & reward log plot
# ******************************

    # 結果のプロット
    plt.plot(np.arange(episode_no), plan_reward)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("plan_value.jpg")
    plt.show()


# ******************************
# write_PSI_data2csv 
# ******************************

    file_name = 'PySI_data_std_IO.csv'         ### .\dataより開きやすい
    #file_name = '.\data\PySI_data_std_IO.csv' 

    write_PSI_data2csv( i_PlanSpace, file_name )


# ******************************
# csv write common_plan_unit.csv 共通計画単位による入出力
# ******************************
# 将来的にSCM treeでサプライチェーン拠点間の需要を連携する時に使用する

    csv_write2common_plan_unit(i_LotSpace,i_PlanSpace, fin_lot_space_Y)


# ******************************
# end of main process
# ******************************
