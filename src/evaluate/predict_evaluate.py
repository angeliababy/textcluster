#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
from sklearn import metrics

# 比较真实结果与预测结果
def get_metrics():
    # dict_lable = {'eco': 1, 'env': 0, 'sports': 2, 'other': 3}
    dict_lable = {'eco': 2, 'env': 0, 'sports': 1, 'other': 3}
    data = pd.read_table(r'id2class.txt', header=None, delim_whitespace=True)
    data2 = pd.read_table(r'cluster_dockmResult.txt', header=None, delim_whitespace=True)
    list_true = []
    list_pred = []
    for i in range(len(data)):
        data.iloc[i, 1] = dict_lable[data.iloc[i, 1]]
        list_true.append(data.iloc[i, 1])
        list_pred.append(data2.iloc[i, 1])

    # 文档链接 http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    #  2.3.10.1 Adjusted Rand index （RI）
    #  2.3.10.2. Mutual Information based scores（NMI）
    #  2.3.10.4. Fowlkes-Mallows scores（FMI）
    #  章节号为文档里面的章节号
    print(metrics.adjusted_rand_score(list_true, list_pred))  # RI指数，越接近1越好
    print(metrics.adjusted_mutual_info_score(list_true, list_pred))  # NMI指数，越接近1越好
    print(metrics.fowlkes_mallows_score(list_true, list_pred))  # FMI指数，越接近1越好


if __name__ == '__main__':

    get_metrics()
