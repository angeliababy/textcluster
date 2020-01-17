# -*- coding: UTF-8 -*-
import os
import re
import pandas as pd
import numpy as np
# 繁体字转简文
from langconv import *


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

# 将所有文本和标题汇总成一个txt
def get_text(base_path, out_path):

    data = pd.read_csv(base_path, usecols=[0, 1, 2], encoding='UTF-8')
    data.columns = ["news_id", "title", "content"]
    # data = open(base_path, 'r', encoding='UTF-8')
    f2 = open(out_path, 'w', encoding='UTF-8')
    for i in range(len(data)):
        try:
            title = data["title"][i].replace('\n', '')
            #
            title = Converter('zh-hans').convert(title)
            title = ''.join(re.findall(u'[\u4e00-\u9fff]+', title))
            content =data["content"][i].replace('\n', '')
            content = ''.join(re.findall(u'[\u4e00-\u9fff]+', content))
            f2.write(str(np.squeeze(data.iloc[i, [0]].values)) + ',')
            f2.write(title+content + '\n')
        except:
            print(data.iloc[i,[0]].values)

    f2.close()

if __name__ == '__main__':

    base_path = "news_title_content.csv"
    out_path = "data.txt"
    get_text(base_path, out_path)


