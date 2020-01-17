#!/usr/bin/python
# -*- coding: UTF-8 -*-
# from sklearn.feature_extraction.text import CountVectorizer
# import lda
import jieba
from gensim import corpora, models
import logging
import traceback
import jieba.posseg as jp
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
import random
import codecs
import sys

# 停用词地址
stopwords_path = "../stopwords.txt"

class TextCluster(object):
    # 初始化函数,重写父类函数
    def __init__(self, stopwords_path = stopwords_path):
        self.stopwords_path = stopwords_path

    # 分词(使用停用词)
    def seg_words(self, sentences, stopwords_path = None):
        if stopwords_path is None:
            stopwords_path = self.stopwords_path

        def stopwordslist(filepath):
            stopwords = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
            return stopwords

        stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径
        title_list = []
        sen_seg_list = []
        for line in sentences:
            if len(line.split(',')) >= 2:
                title_list.append(line.split(',')[0])
                flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
                words = [w.word for w in jp.cut(line.split(',')[1]) if w.flag in flags and w.word not in stopwords and len(w.word)>1]
                sen_seg_list.append(words)
        return title_list, sen_seg_list

    # 加载用户词典
    def load_userdictfile(self, dict_file):
        jieba.load_userdict(dict_file)

    # 读取用户数据
    def load_processfile(self, process_file):
        corpus_list = []
        try:
            fp = open(process_file, "r", encoding='UTF-8')
            for line in fp:
                conline = line.strip()
                corpus_list.append(conline)
            return True, corpus_list
            fp.close()
        except:
            logging.error(traceback.format_exc())
            return False, "get process file fail"

    def evaluate_km(self, tfidf_weight):
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
        for k in range(2, 5):
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
        k = 2 + (Scores.index(max(Scores)))
        return k

    # lda模型，评估num_topics设置主题的个数（聚类无需用）
    def evaluate_lda(self, corpus, dictionary):
        # shuffle corpus洗牌语料库
        cp = list(corpus)
        random.shuffle(cp)

        p = int(len(cp) * .85)
        cp_train = cp[0:p]
        cp_test = cp[p:]

        Perplex = []
        for i in range(15,25):
            lda = models.ldamodel.LdaModel(corpus=cp_train, id2word=dictionary, num_topics=i)
            # Perplexity = lda.log_perplexity(cp_test)
            perplex = lda.bound(cp_test)
            Perplexity = (np.exp2(-perplex / sum(cnt for document in cp_test for _, cnt in document)))

            Perplex.append(Perplexity)

        num_topics = 15 + (Perplex.index(min(Perplex)))
        print(Perplex)
        return num_topics

    # 释放内存资源
    def __del__(self):
        pass

    # 聚类过程
    def process(self, process_file, cluster_ResFileName):
        try:
            # 一、获取标题和分词
            flag, lines = self.load_processfile(process_file)
            if flag == False:
                logging.error("load error")
                return False, "load error"
            # 分词结果与其他方法形式不同
            title_list, sen_seg_list = self.seg_words(lines)

            # # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频(另一种方式)
            # vectorizer = CountVectorizer()
            # x = vectorizer.fit_transform(sen_seg_list)
            # weight = x.toarray()
            #
            # model = lda.LDA(n_topics=5, n_iter=100, random_state=1)
            # model.fit(np.asarray(weight))  # model.fit_transform(X) is also available
            # topic_word = model.topic_word_  # model.components_ also works
            # print(topic_word)
            # # 文档-主题（Document-Topic）分布
            # doc_topic = model.doc_topic_
            # print(doc_topic)
            # # numpy.savetxt('100.csv', doc_topic, delimiter = ',') #将得到的文档-主题分布保存

            # 二、lda模型提取特征
            # 构造词典
            dictionary = corpora.Dictionary(sen_seg_list)
            # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
            corpus = [dictionary.doc2bow(words) for words in sen_seg_list]

            lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=15)

            lda.save('zhwiki_lda.model')
            lda = models.ldamodel.LdaModel.load('zhwiki_lda.model')
            # 打印所有主题，每个主题显示10个词
            for topic in lda.print_topics(num_words=500):
                print(topic)

            # 主题矩阵
            ldainfer = lda.inference(corpus)[0]

            # 主题推断
            print(lda.inference(corpus))
            np.savetxt('100tag.csv', lda.inference(corpus), delimiter=',',fmt='%s')  # 将得到的文档-主题分布保存

            k = self.evaluate_km(ldainfer)
            # 三、Kmeans,大数据量下用Mini-Batch-KMeans算法
            km = KMeans(n_clusters=k)
            km.fit(ldainfer)
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

        except:
            logging.error(traceback.format_exc())
            return False, "process fail"

if __name__ == "__main__":
    # 获取TextProcess对象
    tc = TextCluster(stopwords_path)
    data = sys.argv[1]
    cluster_ResFileName = "cluster_ldakmResulttag1.txt"
    tc.process(data, cluster_ResFileName)
    print("success")

