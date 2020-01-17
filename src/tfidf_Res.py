#!/usr/bin/python
# -*- coding: UTF-8 -*-
import jieba
import logging
import codecs
import traceback
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

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

    # 读取已存在的tfidf权重文件
    def load_tfidf_Res(self, tfidf_ResFileName):
        # tfidf权重已保存
        f = open(tfidf_ResFileName, 'r', encoding='UTF-8')
        line = f.readline()
        line = f.readline()
        tfidf_matrix = []
        while line:
            num = list(map(float, line.split("\t")))
            tfidf_matrix.append(num)
            line = f.readline()
        f.close()
        return tfidf_matrix

    # 输出tfidf矩阵
    def tfidf_Res(self, tfidf_ResFileName, word_list, tfidf_weight):
        tfidf_Res = codecs.open(tfidf_ResFileName, 'w', encoding='UTF-8')
        for num in range(len(word_list)):
            if num == len(word_list) - 1:
                tfidf_Res.write(word_list[num])
            else:
                tfidf_Res.write(word_list[num] + '\t')
        tfidf_Res.write('\r\n')

        # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        for i in range(len(tfidf_weight)):
            for j in range(len(word_list)):
                if j == len(word_list) - 1:
                    tfidf_Res.write(str(tfidf_weight[i][j]))
                else:
                    tfidf_Res.write(str(tfidf_weight[i][j]) + '\t')
            tfidf_Res.write('\r\n')
        tfidf_Res.close()

    # 提取特征
    def process(self, process_file, tfidf_ResFileName):
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

            # 二、tf-idf提取特征
            if not os.path.exists(tfidf_ResFileName):
                # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
                tf_vectorizer = CountVectorizer()

                # fit_transform是将文本转为词频矩阵
                tf_matrix = tf_vectorizer.fit_transform(sen_seg_list)
                # tf_weight = tf_matrix.toarray()
                # print(tf_weight)

                # 该类会统计每个词语的tf-idf权值
                tfidf_transformer = TfidfTransformer()

                # fit_transform是计算tf-idf
                tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)

                # 获取词袋模型中的所有词语
                word_list = tf_vectorizer.get_feature_names()

                # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
                tfidf_weight = tfidf_matrix.toarray()

                self.tfidf_Res(tfidf_ResFileName, word_list, tfidf_weight)

            else:
                tfidf_matrix = self.load_tfidf_Res(tfidf_ResFileName)

            return tfidf_matrix, title_list

        except:
            logging.error(traceback.format_exc())
            return False, "process fail"

    # 释放内存资源
    def __del__(self):
        pass