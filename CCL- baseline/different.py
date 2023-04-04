import openpyxl
import pandas as pd
import numpy as np
from collections import Counter
#-*- coding : utf-8-*-
# coding:unicode_escape
# 读取excel文件
# 括号中的字符串为你要比较的两个excel的路径，注意用“/”
df = pd.DataFrame(pd.read_csv('.\dataset\\activemq.csv', header=None))
df = df.drop(0)
# print(df)
data = df.iloc[:,-4:]
raw_data = data.to_numpy()
# print(raw_data)
new_data = []
for i in range(raw_data.shape[0]):
    a_data = list(map(int, raw_data[i]))
    new_data.append(a_data)
_data = np.array(new_data)

# for i in range(_data.shape[0]):
#     _data[i] = _data[i] + i*10000
# # print(_data)
# data_df = pd.DataFrame(_data)   #关键1，将ndarray格式转换为DataFrame
#
# # 更改表的索引
# data_df.columns = ['A','B','C','D']  #将第一行的0,1,2,...,9变成A,B,C,...,J
#
# # 将文件写入excel表格中
# writer = pd.ExcelWriter('dif.xlsx')
# data_df.to_excel(writer, 'page_1', float_format='%d')
# writer.save()  #关键4

index = []
for i in range(_data.shape[0]):
    temp = 0
    for j in range(_data.shape[1]):
        temp = temp + _data[i][j] * pow(2, j)
    index.append(temp)
b = dict(Counter(index))
print ([key for key,value in b.items()if value > 1])
print ({key:value for key,value in b.items()if value > 1})

# 全检测为clean的有5758
# B-SZZ检测为bug，其他检测为clean的有491
# AG-SZZ检测为bug，其他检测为clean的有38
# MA-SZZ检测为bug，其他检测为clean的有7
# RA-SZZ检测为bug，其他检测为clean的有12
# B-SZZ和AG-SZZ都检测为bug，其他检测为clean的有101
# AG-SZZ和MA-SZZ检测为bug，其他检测为clean的有33
# MA-SZZ和RA-SZZ检测为bug，其他检测为clean的有62
# B-SZZ和MA-SZZ检测为bug，其他检测为clean的有20
# B-SZZ和RA-SZZ检测为bug，其他检测为clean的有16
# AG-SZZ和RA-SZZ检测为bug，其他检测为clean的有0
# 只有B-SZZ检测为clean，其他都检测为bug的有146
# 只有AG-SZZ检测为clean，其他都检测为bug的有230
# 只有MA-SZZ检测为clean，其他都检测为bug的有0
# 只有RA-SZZ检测为clean，其他都检测为bug的有60
# 都检测为bug的有778