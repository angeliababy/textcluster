#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
import codecs
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import tfidf_Res

# 聚类评价方法
def evaluation(tfidf_weight):
    # ### 三者选其一，SSE较好、但需要看拐点，轮廓系数法比较方便
    # # 方法一：'利用SSE选择k（手肘法）'
    # SSE = []  # 存放每次结果的误差平方和
    # for k in range(2, 5):
    #     km = KMeans(n_clusters=k)  # 构造聚类器
    #     km.fit(tfidf_matrix)
    #     SSE.append(km.inertia_)
    #
    # X = range(2, 5)
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # plt.plot(X, SSE, 'o-')
    # plt.show()

    # 方法二：利用轮廓系数法选择k
    Scores = []  # 存放轮廓系数
    for k in range(7, 12):
        km = KMeans(n_clusters=k)  # 构造聚类器
        km.fit(tfidf_weight)
        Scores.append(metrics.silhouette_score(tfidf_weight, km.labels_, metric='euclidean'))

    # X = range(17, 20)
    # plt.xlabel('k')
    # plt.ylabel('轮廓系数')
    # plt.plot(X, Scores, 'o-')
    # plt.show()

    # # 方法三：值越大越好，重点是速度快
    # Ss = []  # 存放Ss
    # for k in range(12, 30, 2):
    #     km = KMeans(n_clusters=k)  # 构造聚类器
    #     km.fit(tfidf_weight)
    #     Ss.append(metrics.calinski_harabaz_score(tfidf_weight, km.labels_))
    #
    # X = range(2, 5)
    # plt.xlabel('k')
    # plt.ylabel('Ss')
    # plt.plot(X, Ss, 'o-')
    # plt.show()

    # 求最优k值
    print(Scores)
    k = 7 + (Scores.index(max(Scores)))

    return k

# 聚类过程
def kmeans(tfidf_matrix, title_list, cluster_ResFileName):
    k = evaluation(tfidf_matrix)
    # 三、Kmeans,大数据量下用Mini-Batch-KMeans算法
    km = KMeans(n_clusters=k)
    km.fit(tfidf_matrix)
    print(Counter(km.labels_))  # 打印每个类多少个
    # print(km.cluster_centers_)   # 中心点

    # 存储每个样本所属的簇
    clusterRes = codecs.open(cluster_ResFileName, 'w', encoding='UTF-8')
    count = 1
    while count <= len(km.labels_):
        clusterRes.write(str(title_list[count - 1]) + '\t' + str(km.labels_[count - 1]))
        clusterRes.write('\r\n')
        count = count + 1
    clusterRes.close()

    # # 四、可视化
    # # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    # tsne = TSNE(n_components=2)
    # decomposition_data = tsne.fit_transform(tfidf_weight)
    #
    # x = []
    # y = []
    # for i in decomposition_data:
    #     x.append(i[0])
    #     y.append(i[1])
    #
    # plt.scatter(x, y, c=km.labels_)
    # plt.show()
    # plt.savefig('./sample.png', aspect=1)

# 类似于主函数
if __name__ == "__main__":
    # 获取TextProcess对象
    tc = tfidf_Res.TextCluster()

    data = sys.argv[1]
    tfidf_matrix, title_list = tc.process(data, "tfidf_Resulttag.txt")
    print("success load tfidf_matrix")
    cluster_ResFileName = "cluster_idfkmResulttag.txt"
    kmeans(tfidf_matrix, title_list, cluster_ResFileName)
    print("success")