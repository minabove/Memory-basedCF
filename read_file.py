import numpy as np

def read_train_test():
    n = 943
    m = 1682
    train_data = np.zeros((n, m), dtype=np.int)  # 从文件中读入base 二维数组 初始化为0
    test_data = np.zeros((n, m), dtype=np.int)  # 从文件中读入test 二维数组，初始化为0

    file_base = ''
    file_test = ''

    try:
        file_base = open('u1.base', 'r')
        for line in file_base.readlines():
            line = line.strip()
            ss = line.split()
            row = int(ss[0])
            col = int(ss[1])
            r = int(ss[2])
            train_data[row - 1][col - 1] = r
        file_test = open('u1.test', 'r')
        for line in file_test.readlines():
            line = line.strip()
            ss = line.split()
            row = int(ss[0])
            col = int(ss[1])
            r = int(ss[2])
            test_data[row - 1][col - 1] = r
    finally:
        if file_base:
            file_base.close()
        if file_test:
            file_test.close()

    return (train_data, test_data)

def read_Suw():
    n = 943
    Swu = np.zeros((n,n), dtype=float)           # 不同用户之间的相似度二维数组
    Swu_file = ' '
    try:
        Swu_file = open('Swu.txt', 'r')
        for line in Swu_file.readlines():
            line = line.strip()
            ss = line.split()
            row = int(ss[0])
            col = int(ss[1])
            r = float(ss[2])
            Swu[row][col] = r
    finally:
        if Swu_file:
            Swu_file.close()

    return Swu

def read_Skj():
    m = 1682
    Skj = np.zeros((m,m), dtype=float)
    Skj_file = ' '
    try:
        Skj_file = open('Skj.txt', 'r')
        for line in Skj_file.readlines():
            line = line.strip()
            ss = line.split()
            row = int(ss[0])
            col = int(ss[1])
            r = float(ss[2])
            Skj[row][col] = r
            Skj[col][row] = r
    finally:
        if Skj_file:
            Skj_file.close()

    return Skj





