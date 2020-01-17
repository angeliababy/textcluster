# -*- coding: UTF-8 -*-
import os
import re

# 文章列表
# 将所有文本和标题汇总成一个txt
def get_text(base_path, out_path):

    filelist = os.listdir(base_path)
    f2 = open(out_path, 'w', encoding='UTF-8')
    for files in filelist:
        # print (files)
        filename = files.split('.')[0]
        f = open(base_path + files, 'r', encoding='UTF-8')
        text = f.read().replace('\n', '')

        data = ''.join(re.findall(u'[\u4e00-\u9fff]+', text))  # 必须为unicode类型，取出所有中文字符
        f2.write(filename + ',')
        f2.write(data + '\n')
    f2.close()

if __name__ == '__main__':

    base_path = "../../data_offline/"
    out_path = "data.txt"
    get_text(base_path, out_path)


