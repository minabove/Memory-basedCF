import read_file as rf
import computing as cp

data = rf.read_train_test()           # 从文件中读取 u1.base  u1.test  得到两个二维数组

train_data = data[0]
test_data = data[1]

# 计算 各个用户 的本身的所有打分的平均值  得到 aver_u_to_i 列表
aver_u_to_i = cp.compute_aver_u_to_i(train_data)

# 计算 两两用户 之间的 相似度 二维数组
cp.compute_Swu(train_data, aver_u_to_i)


