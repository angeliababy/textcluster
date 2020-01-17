# 文本特征处理及聚类的几种方法
本项目完整源码地址：
[https://github.com/angeliababy/textcluster](https://github.com/angeliababy/textcluster)

项目博客地址: 
[https://blog.csdn.net/qq_29153321/article/details/104015257](https://blog.csdn.net/qq_29153321/article/details/104015257)
## 数据准备

###   测试数据说明

data_offline文件夹包含200 economy 类，200个sports类，200个environment类，50个other类，为线下做试验的数据集，id2class.txt为data_offline文件夹中每个文件对应的类别，以此可以比较聚类效果。

src/get_data下为数据准备的过程，获取去除标点符号的文本及编号：

1.get_res.py为处理普通的txt文件
```
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
```
2.get_res_csv.py为处理csv文件，并将繁体文转化为简体文
```
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
```

## 文本特征处理
**1.tf-idf:**
tf-idf = tf(词频)*idf(逆词频)
其中idf(x) = log(N/N(x))
tfidf_Res.py:
```
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
```

**2.doc2vec**
句向量，是 word2vec 的拓展，考虑了句子的序号id
```
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
```

**3.lda**
主题模型，词袋模型，完全考虑词语的分布来判断其主题分布，并依据每个文本的主题概率分布来进行聚类
```
# 一、获取标题和分词
flag, lines = self.load_processfile(process_file)
if flag == False:
    logging.error("load error")
    return False, "load error"
# 分词结果与其他方法形式不同
title_list, sen_seg_list = self.seg_words(lines)

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
```

## 聚类分析
kmeans:
k簇中心,离簇中心最近进行聚类
```
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
```
聚类个数选择：
```
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
```

dbscan:
基于密度的聚类，聚类的时候不需要预先指定簇的个数
```
# DBSCAN参数不好调节
db = DBSCAN(eps=1.25, min_samples=10).fit(tfidf_weight)
print(db.core_sample_indices_)
print(db.labels_)
# 聚类个数为1-n之间会报错
Score = metrics.silhouette_score(tfidf_weight, db.labels_)
print(Score)

# 存储每个样本所属的簇
clusterRes = codecs.open(cluster_ResFileName, 'w', encoding='UTF-8')
count = 1
while count <= len(db.labels_):
    clusterRes.write(str(title_list[count - 1]) + '\t' + str(db.labels_[count - 1]))
    clusterRes.write('\r\n')
    count = count + 1
clusterRes.close()
```

gsdmm:
可以很好的处理稀疏、高纬度的短文本
```
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
```

模型评估
src/evaluate/predict_evaluate.py
```
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
```


参考数据集：

data_offline:

Total:650(other+3分类)


Idfkm:4 (分类2-4)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020011710315612.png)



Doc2veckm:3 (分类2-4)较慢

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117103215185.png)


Idfdb:-1和3

很差，-1为离群点，254个离群点，仅21个为other


Idfdsgmm:9(10)

很慢，只有1类，很差


Ldakm

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117103239653.png)
测试集中doc2vec+kmeans效果最好，tf-idf+kmeans其次		
