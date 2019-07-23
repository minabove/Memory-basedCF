import numpy as np

def hybrid_cf(pre_ruj_ucf, pre_ruj_icf, test_data):
    param = 0.5

    n = 943
    m = 1682

    pre_ruj_hybrid = np.zeros((n, m), dtype=np.float)

    for u in range(n):
        for j in range(m):
            pre_ruj_hybrid[u][j] = param * pre_ruj_ucf[u][j] + (1-param) * pre_ruj_icf[u][j]

    test_r_num = 20000
    numerator = 0.0

    # 计算 Hybrid_CF 的 RMSE
    for u in range(n):
        for i in range(m):
            if test_data[u][i] == 0:
                continue
            numerator += (test_data[u][i] - pre_ruj_hybrid[u][i]) * (test_data[u][i] - pre_ruj_hybrid[u][i])

    numerator /= test_r_num
    RMSE = np.sqrt(numerator)

    numerator = 0.0
    # 计算 Hybrid_CF 的 MAE
    for u in range(n):
        for i in range(m):
            if test_data[u][i] == 0:
                continue
            numerator += np.fabs(test_data[u][i] - pre_ruj_hybrid[u][i])

    MAE = numerator / test_r_num

    print('Hybrid-based CF:')
    print('RMSE: ', round(RMSE, 4), end='   ')
    print('MAE: ', round(MAE, 4))