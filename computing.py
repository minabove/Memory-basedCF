import numpy as np

# 计算各个用户打分的平均值
def compute_aver_u_to_i(train_data):
    n = 943
    train_r_num = 80000
    aver_all = train_data.sum()/train_r_num         # 全部评分的平均值

    aver_u_to_i = np.zeros(n, dtype=float)          # 计算各个用户打分的平均值

    for i in range(train_data.shape[0]):
        numerator = 0.0
        denominator = 0.0

        for j in range(train_data.shape[1]):
            if train_data[i][j] > 0:
                numerator += train_data[i][j]
                denominator += 1

        if numerator==0 or denominator==0:
            aver_u_to_i[i] = aver_all
            continue

        aver_u_to_i[i] = numerator/denominator

    return aver_u_to_i

# 计算两两用户之间的相似度
def compute_Swu(train_data, aver_u_i):
    # 将 Swu 写入文件
    Swu_file = open('Swu.txt', 'w')

    n = 943         # 用户个数
    Swu = np.zeros((n,n), dtype=np.float)           # 不同用户之间的相似度二维数组

    for u in range(n):
        for w in range(u+1, n):
            # 获取 用户u 和用户w评分的非零值的索引列表
            nonzero_U_index = [i for i,e in enumerate(train_data[u]) if e!=0]
            nonzero_W_index = [i for i,e in enumerate(train_data[w]) if e!=0]

            # 取交集 得到 用户u和w 都有评价的物品 的索引 的集合
            intersection = set(nonzero_U_index) & set(nonzero_W_index)
            # 再转化为list
            common = [i for i in intersection]

            numerator = 0.0
            num1 = 0.0
            num2 = 0.0

            for k in common:
                numerator = numerator + (train_data[u][k] - aver_u_i[u]) * (train_data[w][k] - aver_u_i[w])
                num1 += (train_data[u][k] - aver_u_i[u]) * (train_data[u][k] - aver_u_i[u])
                num2 += (train_data[w][k] - aver_u_i[w]) * (train_data[w][k] - aver_u_i[w])

            denominator = np.sqrt(num1) * np.sqrt(num2)

            if denominator == 0 or numerator == 0:
                Swu[w][u] = Swu[u][w] = 0
                Swu_file.write(str(w) + " " + str(u) + " " + str(Swu[w][u]) + "\n")
                Swu_file.write(str(u) + " " + str(w) + " " + str(Swu[w][u]) + "\n")
                continue

            Swu[w][u] = numerator / denominator
            Swu[u][w] = Swu[w][u]

            Swu_file.write(str(w) + " " + str(u) + " " + str(Swu[w][u]) + "\n")
            Swu_file.write(str(u) + " " + str(w) + " " + str(Swu[w][u]) + "\n")

    Swu_file.close()

# 计算两两物品之间的相似度
def compute_Skj(train_data, aver_u_i):
    # Skj 写入文件
    Skj_file = open('Skj.txt', 'w')

    n = 943
    m = 1682
    Skj = np.zeros((m,m), dtype=np.float)

    for k in range(m):
        for j in range(k+1, m):
            # 分别求得 有对 物品k 和 物品j的用户编号集合
            nonzero_K_index = []
            nonzero_J_index = []
            for i in range(n):
                if train_data[i][k] != 0:
                    nonzero_K_index.append(i)
                if train_data[i][j] != 0:
                    nonzero_J_index.append(i)

            intersection = set(nonzero_J_index) & set(nonzero_K_index)
            common = [i for i in intersection]

          #  print(common)

            numerator = 0.0
            num1 = 0.0
            num2 = 0.0

            for u in common:
                numerator += (train_data[u][k] - aver_u_i[u]) * (train_data[u][j] - aver_u_i[u])
                num1 += (train_data[u][k] - aver_u_i[u]) * (train_data[u][k] - aver_u_i[u])
                num2 += (train_data[u][j] - aver_u_i[u]) * (train_data[u][j] - aver_u_i[u])

            denominator = np.sqrt(num1) * np.sqrt(num2)

            if denominator == 0 or numerator == 0:
                Skj[k][j] = Skj[j][k] = 0
                Skj_file.write(str(k) + " " + str(j) + " " + str(Skj[k][j]) + "\n")
               # print(Skj[k][j])
                continue

            Skj[k][j] = numerator / denominator

            if Skj[k][j] > 1.0:
                Skj[k][j] = 1.0
            if Skj[k][j] < -1.0:
                Skj[k][j] = -1.0

            Skj[j][k] = Skj[k][j]
            Skj_file.write(str(k) + " " + str(j) + " " + str(Skj[k][j]) + "\n")
         #   print(Skj[k][j])

    Skj_file.close()












