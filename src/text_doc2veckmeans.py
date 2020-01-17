# -*- coding: UTF-8 -*-
import jieba
import logging
import codecs
import traceback
import gensim
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
import os
import sys
import matplotlib.pyplot as plt


# 停用词地址
stopwords_path = "../stopwords.txt"

class TextCluster(object):
    # 初始化函数,重写父类函数
    def __init__(self, stopwords_path = stopwords_path):
        self.stopwords_path = stopwords_path

    # # 分词(不使用停用词)
    # def seg_words(self, sentence):
    #     seg_list = jieba.cut(sentence)  # 默认是精确模式
    #     return " ".join(seg_list)      # 分词，然后将结果列表形式转换为字符串

    # 分词(使用停用词)
    def seg_words(self, sentence, stopwords_path = None):
        if stopwords_path is None:
            stopwords_path = self.stopwords_path

        def stopwordslist(filepath):
            stopwords = [line.strip() for line in open(filepath, 'r', encoding='UTF-8').readlines()]
            return stopwords

        sentence_seged = jieba.cut(sentence.strip())
        stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径
        outstr = ''  # 返回值是字符串
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

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
            fp.close()
            return True, corpus_list
        except:
            logging.error(traceback.format_exc())
            return False, "get process file fail"

    # 存储分词文本（不含标题）
    def output_file(self, out_file, sen_seg_list):

        try:
            data1 = codecs.open(out_file, 'w', encoding='UTF-8')
            for num in range(len(sen_seg_list)):
                if num == len(sen_seg_list) - 1:
                    data1.write(sen_seg_list[num])
                else:
                    data1.write(sen_seg_list[num] + '\r\n')
            data1.close()
        except:
            logging.error(traceback.format_exc())
            return False, "out file fail"

    # 评价算法好坏
    def evaluation(self, tfidf_matrix):
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
        for k in range(15, 25):
            km = KMeans(n_clusters=k)  # 构造聚类器
            km.fit(tfidf_matrix)
            Scores.append(metrics.silhouette_score(tfidf_matrix, km.labels_, metric='euclidean'))

        # X = range(2, 5)
        # plt.xlabel('k')
        # plt.ylabel('轮廓系数')
        # plt.plot(X, Scores, 'o-')
        # plt.show()

        # 方法三：值越大越好，重点是速度快
        # Ss = []  # 存放Ss
        # for k in range(15, 30, 2):
        #     km = KMeans(n_clusters=k)  # 构造聚类器
        #     km.fit(tfidf_matrix)
        #     Ss.append(metrics.calinski_harabaz_score(tfidf_matrix, km.labels_))
        #
        # X = range(2, 5)
        # plt.xlabel('k')
        # plt.ylabel('Ss')
        # plt.plot(X, Ss, 'o-')
        # plt.show()

        # 求最优k值
        print(Scores)
        k = 15 + (Scores.index(max(Scores)))

        return k

    # 释放内存资源
    def __del__(self):
        pass

    # 聚类过程
    def process(self, process_file, num_clusters, cluster_ResFileName, data1, modelpath):
        try:
            # 一、获取标题和分词
            sen_seg_list = []
            title_list = []
            flag, lines = self.load_processfile(process_file)
            if flag == False:
                logging.error("load error")
                return False, "load error"
            for line in lines:
                title_list.append(line.split(',')[0])
                sen_seg_list.append(self.seg_words(line.split(',')[1]))

            if not os.path.exists(modelpath):
                # 存储分词文本
                if not os.path.exists(data1):
                    self.output_file(data1, sen_seg_list)
                    print("success output")

            # doc2vec提取特征
            sentences = gensim.models.doc2vec.TaggedLineDocument(data1)

            if not os.path.exists(modelpath):
                # doc2vec提取特征
                # 训练并保存模型
                model = gensim.models.Doc2Vec(sentences, size=100, window=2, min_count=3)
                model.train(sentences, total_examples=model.corpus_count, epochs=1000)
                model.save(modelpath)

            infered_vectors_list = []
            print("load doc2vec model...")
            model_dm = gensim.models.Doc2Vec.load(modelpath)
            print("load train vectors...")
            i = 0
            for text, label in sentences:
                vector = model_dm.infer_vector(text)
                infered_vectors_list.append(vector)
                i += 1

            k = self.evaluation(infered_vectors_list)
            # Kmeans,大数据量下用Mini-Batch-KMeans算法
            km = KMeans(n_clusters=k)

            # 可直接用模型
            # km.fit(model.docvecs)
            km.fit(infered_vectors_list)
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


# 类似于主函数
if __name__ == "__main__":
    # 获取TextProcess对象
    tc = TextCluster(stopwords_path)

    data = sys.argv[1]
    tc.process(data, 3, "cluster_dockmResulttag.txt", "get_data/data1tag.txt", "../model/demoDoc2Vectag.pkl")
    print("success")