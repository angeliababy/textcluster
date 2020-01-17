#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 适合短文本（很慢）
import codecs
from gmp import MovieGroupProcess
import sys

import tfidf_Res


# 聚类过程
def gsdmm(tfidf_matrix, title_list, cluster_ResFileName):
    # GSDMM文本聚类
    mgp = MovieGroupProcess(K=35, alpha=0.1, beta=0.1, n_iters=20)
    y = mgp.fit(tfidf_matrix, len(tfidf_matrix))
    print(y)

    # 存储每个样本所属的簇
    clusterRes = codecs.open(cluster_ResFileName, 'w', encoding='UTF-8')
    count = 1
    while count <= len(y):
        clusterRes.write(str(title_list[count - 1]) + '\t' + str(y[count - 1]))
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
    cluster_ResFileName = "cluster_idfgmResulttag.txt"
    gsdmm(tfidf_matrix, title_list, cluster_ResFileName)
    print("success")