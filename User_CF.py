import read_file as rf
#import computing as cp
import numpy as np

def user_based_CF(train_data, test_data, aver_u_to_i):
    n = 943
    m = 1682

    Swu = rf.read_Suw()

    # 对每个用户 寻找 k 个 最近邻居 也就是 相似度最高的 K个
    K = 50

    predict_ruj = np.zeros((n, m), dtype=np.float)

    # 对每个用户 求出 Nuj
    # 先求出各个用户的 Nu 即为 每个用户筛去 与其Swu 为0 的用户后剩下的用户集合
    Nu = {}
    for u in range(n):
        nu = []
        for w in range(n):
            if Swu[u][w] != 0:
                nu.append(w)
        Nu[u] = nu

    # 然后求出 对物品 j 有评分的用户集
    Uj = {}
    for j in range(m):
        uj = []
        for u in range(n):
            if train_data[u][j] != 0:
                uj.append(u)
        Uj[j] = uj

    # 将 Nu 和 Uj 中的元素 两两组合，并求交集 取Swu 最大的前K个 得到Nuj
    for u in range(n):
        for j in range(m):
            intersection = set(Nu[u]) & set(Uj[j])
            common = [i for i in intersection]
            # common 是list 是用户编号的list 要依据这个list找到Swu最大的K个用户

            dic_Nuj = {}
            for i in common:
                dic_Nuj[i] = Swu[u][i]

            # 排序后的返回值是一个list，而原字典中的名值对被转换为了list中的元组
            a = sorted(dic_Nuj.items(), key=lambda x: x[1], reverse=True)
            # eg. [(3, 38), (4, 21), (1, 11), (2, 2)]

            Nuj = []
            counter = 0
            for tp in a:
                Nuj.append(tp[0])
                counter = counter + 1
                if counter >= K:
                    break

            if Nuj:
                # 对于用户 u 和物品 j，已经得到了Nuj 即K个最相似的用户的集合，可以计算相应的predict_ruj
                predict_ruj[u][j] += aver_u_to_i[u]
                numerator = 0.0
                denominator = 0.0
                for w in Nuj:
                    numerator += Swu[u][w] * (train_data[w][j] - aver_u_to_i[w])
                    denominator += + np.fabs(Swu[u][w])

                predict_ruj[u][j] += numerator/denominator
            else:
                predict_ruj[u][j] = aver_u_to_i[u]

            if predict_ruj[u][j] > 5:
                predict_ruj[u][j] = 5
            if predict_ruj[u][j] < 1:
                predict_ruj[u][j] = 1

    test_r_num = 20000
    numerator = 0.0

    # 计算User-based CF 的 RMSE
    for u in range(n):
        for i in range(m):
            if test_data[u][i] == 0:
                continue
            numerator += (test_data[u][i] - predict_ruj[u][i])*(test_data[u][i] - predict_ruj[u][i])

    numerator /= test_r_num
    RMSE = np.sqrt(numerator)

    numerator = 0.0
    # 计算User-based CF 的 MAE
    for u in range(n):
        for i in range(m):
            if test_data[u][i] == 0:
                continue
            numerator += np.fabs(test_data[u][i] - predict_ruj[u][i])

    MAE = numerator / test_r_num

    print('User-based CF:')
    print('RMSE: ', round(RMSE, 4), end='   ')
    print('MAE: ', round(MAE, 4))

    return predict_ruj














