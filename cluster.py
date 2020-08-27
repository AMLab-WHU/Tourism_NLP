import re
import jieba
import numpy as np
import pandas as pd
from jieba import analyse
import jieba.posseg as pseg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer  # 基于TF-IDF的词频转向量库
from sklearn.cluster import KMeans
from data_admin import Database

# 中文分词
def seg_sentence(text):
    # sentence_seged = jieba.cut(text)
    word_list = []  # 建立空列表用于存储分词结果
    seg_list = pseg.cut(text)  # 精确模式分词[默认模式]
    for word in seg_list:
        if word.flag in ['n', 'ns', 'nt', 'nz', 'nrt', 'nrfg', 'nr', 'ng']:  # 选择属性
            word_list.append(word.word)  # 分词追加到列表
    stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]# 这里加载停用词的路径
    outstr = ''
    for word in word_list:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

# 数据清理
def data_clean(text):
    text = re.sub(r'\n+', '', text)  # remove \n，re.sub(r'a','b','c'),将c中的a部分替换为b
    text = re.sub(r' +', '', text)  # remove blank        content[i] = re.sub(r'\W+', ' ', content[i])  # 用空格替换特殊字符，即评论中的表情等
    remove_list = ['x0A', '景区', '九寨沟', '九寨']  # 要删除的词的列表
    for re_wd in remove_list:
        text = re.sub(re_wd, ' ', text)  # replace symbols with blank
    return text

# 计算tf-idf，提取关键词列表
def getKeywords_tfidf(text,num):
    line_seg = seg_sentence(text)
    key_words_TFIDF = analyse.extract_tags(line_seg, topK=num) # 提取每一句评论的前num个关键词
    kwds = " ".join(key_words_TFIDF)
    return kwds

# 词向量模型
def vectorizer(kwds_list):
    vectorizer = TfidfVectorizer()  # 创建词向量模型
    X = vectorizer.fit_transform(kwds_list)  # 将评论关键字列表转换为词向量空间模型
    word_vectors = vectorizer.get_feature_names()  # 词向量
    # print(word_vectors)
    word_values = X.toarray()  # 向量值
    return word_values

# K均值聚类
def cluster(num, kwds_list):
    word_values = vectorizer(kwds_list)
    down_num = 10 # 降维维数
    # 数据框标签
    column = []
    i = 1
    while i <= down_num:
        column.append('pca_'+str(i))
        i += 1
    # PCA降维
    pca = PCA(n_components=down_num)  # 输出两维
    newData = pca.fit_transform(word_values)  # 载入N维
    # K均值聚类
    model_kmeans = KMeans(n_clusters=num)  # 创建聚类模型对象,num指定类簇数量
    model_kmeans.fit(newData)  # 训练模型
    # 聚类结果汇总
    cluster_labels = model_kmeans.labels_  # 聚类标签结果
    column.append('cluster_labels')
    column.append('keywords')
    kwds_list = np.array(kwds_list)
    comment_matrix = np.hstack((newData, cluster_labels.reshape(newData.shape[0], 1), kwds_list.reshape(newData.shape[0], 1)))  # 将向量值和标签值合并为新的矩阵
    comment_pd = pd.DataFrame(comment_matrix, columns=column)  # 创建包含词向量和聚类标签的数据框
    #word_vectors.append('cluster_labels')  # 将新的聚类标签列表追加到词向量后面
    #comment_pd = pd.DataFrame(comment_matrix, columns=word_vectors)  # 创建包含词向量和聚类标签的数据框
    comment_pd.to_csv('comment.csv')

    # comment_matrix = np.hstack((newData, cluster_labels.reshape(newData.shape[0], 1)))  # 将向量值和标签值合并为新的矩阵
    # comment_pd = pd.DataFrame(comment_matrix, columns=['pca_x','pca_y','cluster_labels'])  # 创建包含词向量和聚类标签的数据框
    # comment_pd.to_csv('pca.csv')
    # cluster_label = cluster_labels.reshape(newData.shape[0], 1)
    # print(cluster_label)
    # return cluster_label, newData
    return comment_pd

# 输出每一类中句子的关键词
def print_cluster(num, comment_pd):
    i = 0
    while i < num:
        data_p = comment_pd[(comment_pd.cluster_labels == str(i))]['keywords']
        data_p = np.array(data_p)
        print('类别',i,'：')
        print(data_p)
        i+=1

def draw_cluster(label, reduced_x):
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    yellow_x, yellow_y = [], []
    black_x, black_y = [], []
    for i in range(len(reduced_x)):
        if label[i][0] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif label[i][0] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        elif label[i][0] == 2:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
        elif label[i][0] == 3:
            yellow_x.append(reduced_x[i][0])
            yellow_y.append(reduced_x[i][1])
        else:
            black_x.append(reduced_x[i][0])
            black_y.append(reduced_x[i][1])
    # 可视化
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.scatter(yellow_x, yellow_y, c='y', marker='>')
    plt.scatter(black_x, black_y, c='k', marker='<')
    plt.show()

def main():
    Data_admin = Database()
    sql_1 = 'select autoId, content from sentiment'
    data_1 = Data_admin.database(sql_1)
    kwds_list = [] # 每句评论的关键词列表
    # content_list = []
    for text in data_1:
        item = data_clean(text[1]) # 数据清理
        # content_list.append(seg_sentence(item))
        kwds_list.append(getKeywords_tfidf(item,5)) #每句评论提取前5个关键词
    # label, reduced_x = cluster(5, kwds_list)  #设置类簇数量为5
    comment_pd = cluster(5, kwds_list)  #设置类簇数量为5
    print_cluster(5, comment_pd) #设置类簇数量为5
    # draw_cluster(label, reduced_x)

if __name__ == '__main__':
    main()