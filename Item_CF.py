import read_file as rf
import numpy as np

def item_based_CF(train_data, test_data, aver_u_i):
    #data = rf.read_train_test()
    n = 943
    m = 1682
    K = 50

    # prediction
    pred_ruj = np.zeros((n, m), dtype=np.float)

    Skj = rf.read_Skj()

    # 求出 物品 j 的邻居集合 Nj   先筛去所有相似度为0的物品
    Nj = {}
    for j in range(m):
        nj = []
        for k in range(m):
            if Skj[j][k] > 0:
            # 刚开始时写成了Skj[j][k]!=0 由于有很多的Skj 为-1 导致后面误差很大!!! 浪费了两个小时
                nj.append(k)
        Nj[j] = nj

    # 求出 被 用户 u 评分过的物品 的 集合
    Iu = {}
    for u in range(n):
        iu = []
        for i in range(m):
            if train_data[u][i] != 0:
                iu.append(i)
        Iu[u] = iu

    # 将 Nj 和 Iu  两两组合 求出 交集  再取相似度最大的K个 得到 Nju
    for u in range(n):
        for j in range(m):
            intersection = set(Nj[j]) & set(Iu[u])
            common = [i for i in intersection]

            dic_Nju = {}
            for i in common:
                dic_Nju[i] = Skj[j][i]

            a = sorted(dic_Nju.items(), key=lambda x:x[1], reverse=True)

            # 实在没找到错误的时候，尝试了以下的方法
            '''
            if len(a) >= 50:
                K = 50
            elif len(a) >= 40:
                K = 40
            elif len(a) >= 30:
                K = 30
            elif len(a) >= 20:
                K = 20
            elif len(a) >= 10:
                K = 10
            else:
                K = 5
            '''

            Nju = []
            counter = 0
            for tp in a:
                Nju.append(tp[0])
                counter += 1
                if counter >= K:
                    break

            if Nju:
                numerator = 0.0
                denominator = 0.0
                for k in Nju:
                    if Skj[k][j] == 0 or train_data[u][k] == 0:
                        continue
                    numerator += Skj[k][j] * train_data[u][k]
                    denominator += Skj[k][j]

                pred_ruj[u][j] = numerator / denominator
            else:
                pred_ruj[u][j] = aver_u_i[u]

            if pred_ruj[u][j] > 5:
                pred_ruj[u][j] = 5
            if pred_ruj[u][j] < 1:
                pred_ruj[u][j] = 1

    test_r_num = 20000
    numerator = 0.0

    # 计算Item-based CF 的 RMSE
    for u in range(n):
        for i in range(m):
            if test_data[u][i] == 0:
                continue
            numerator += (test_data[u][i] - pred_ruj[u][i])*(test_data[u][i] - pred_ruj[u][i])

    numerator /= test_r_num
    RMSE = np.sqrt(numerator)

    numerator = 0.0
    # 计算Item-based CF 的 MAE
    for u in range(n):
        for i in range(m):
            if test_data[u][i] == 0:
                continue
            numerator += np.fabs(test_data[u][i] - pred_ruj[u][i])

    MAE = numerator / test_r_num

    print('Item-based CF:')
    print('RMSE: ', round(RMSE, 4), end='   ')
    print('MAE: ', round(MAE, 4))

    return pred_ruj










