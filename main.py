import Item_CF as icf
import User_CF as ucf
import read_file as rf
import computing as cp
import Hybrid_CF as hcf

data = rf.read_train_test()           # 从文件中读取 u1.base  u1.test  得到两个二维数组

train_data = data[0]
test_data = data[1]

# 计算 各个用户 的本身的所有打分的平均值  得到 aver_u_to_i 列表
aver_u_i = cp.compute_aver_u_to_i(train_data)

# user_based_CF  求得 prediction_ruj 并输出 RMSE, MAE
pre_ruj_ucf = ucf.user_based_CF(train_data, test_data, aver_u_i)

# item_based_CF  求得 prediction_ruj 并输出 RMSE, MAE
pre_ruj_icf = icf.item_based_CF(train_data, test_data, aver_u_i)

hcf.hybrid_cf(pre_ruj_ucf, pre_ruj_icf, test_data)