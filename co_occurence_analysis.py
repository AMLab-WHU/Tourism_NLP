# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import networkx as nx
import matplotlib.pyplot as plt
import codecs
# 导入系统模块
import sys
# imp模块提供了一个可以实现import语句的接口
from imp import reload
# import pymysql

# import sys;
# sys.path.append('/Users/co-occurence1');
# from Database import Database;
from data_admin import Database


# 定义分词函数
def tokenizer(s, stopwords):
    words = []
    cut = jieba.cut(s)
    for word in cut:
        if word not in stopwords:
            words.append(word)
    return words


# 定义分词函数，以进行tfidf矩阵计算
def tokenizer_tfidf(s, stopwords):
    words = []
    cut = jieba.cut(s)
    for word in cut:
        if word not in stopwords:
            words.append(word)
    return words


# 格式化需要计算的数据，将原始数据格式转换成二维数组
def format_data(comment_list, set_key_list, stopwords):
    commentwords_list = []
    '''
    for i in range(len(comment_list)):
        commentwords_list.append([])
        i+1
    '''
    for i in range(len(comment_list)):
        words = tokenizer(comment_list[i], stopwords)
        # intersection = []
        intersection = [i for i in words if i in set_key_list]
        intersection = list(set(filter(lambda x: x != '', intersection)))
        # print(type(words))
        if intersection:
            commentwords_list.append(intersection)

    for i in range(len(commentwords_list)):
        print(commentwords_list[i])

    return commentwords_list


# 格式化数据，将原始数据格式转换成一维数组（示例：["小明 硕士 毕业 与 中国 科学院","我 爱 北京 天安门"]）
# 每条评论的分词结果不进行去重处理（计算tf值）
def format_data_duplication(comment_list, stopwords):
    commentwords_list_duplication = []
    for i in range(len(comment_list)):
        words = tokenizer(comment_list[i], stopwords)
        text = " ".join(words)
        commentwords_list_duplication.append(text)

    # for i in range(len(comment_list)):
    # print(commentwords_list_duplication[i])
    # print(len(commentwords_list_duplication))
    return commentwords_list_duplication


# 构建词频字典
def word_frequency_dict(key_list):
    frequency_dict = {}
    for i in range(len(key_list)):
        if key_list[i] in frequency_dict:
            count = frequency_dict[key_list[i]]
        else:
            count = 0
        count = count + 1
        frequency_dict[key_list[i]] = count
    return frequency_dict


# print(word_frequency_dict(key_list))
# print(sorted(word_frequency_dict(key_list).items(), key=lambda item: item[1], reverse=True))


# tf-idf获取文本top10关键词
"""
       TF-IDF权重：
           1、CountVectorizer 构建词频矩阵
           2、TfidfTransformer 构建tfidf权值计算
           3、文本的关键字
           4、对应的tfidf矩阵
"""


def getKeywords_tfidf(corpus, topK):
    # idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    # corpus = [] # 将所有文档输出到一个list中，一行就是一个文档

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)  # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 5、打印词语权重
    # ids, titles, keys = [], [], []
    keys = []
    for i in range(len(weight)):
        # print(u"-------这里输出第", i+1 , u"篇文本的词语tf-idf------")
        # ids.append(idList[i])
        # titles.append(titleList[i])
        df_word, df_weight = [], []  # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            # print(word[j],weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1)  # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight", ascending=False)  # 按照权重值降序排列
        keyword = np.array(word_weight['word'])  # 选择词汇列并转成数组格式
        word_split = [keyword[x] for x in range(0, topK)]  # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        # word_split1 = word_split.encode('raw_unicode_escape')
        # word_split2 =word_split1.decode()
        # keys.append(word_split2.encode("utf-8"))
        keys.append(word_split)

    result = pd.DataFrame({"key": keys}, columns=['key'])
    # print("bbbbbbbbbbbb")
    print(result)
    return result


# 构建词频字典
def tfidf_dict(commentwords_list, commentwords_list_duplication, key_list):
    appeardict = {}  # 每个关键词与 [出现在的行(formated_data)的list] 组成的dictionary
    for w in key_list:  # keylist：所有关键词
        appearlist = []
        i = 0
        for each_line in commentwords_list:
            if w in each_line:
                appearlist.append(i)
            i += 1
        appeardict[w] = appearlist

    for i in range(len(commentwords_list_duplication)):
        # a = 0
        pass
    # tf = word_frequency_dict(key_list)
    return word_frequency_dict(key_list)  # xts 删除重复代码，调用上面的函数


# print(word_frequency_dict(key_list))
# print(sorted(word_frequency_dict(key_list).items(), key=lambda item: item[1], reverse=True))


# 选取频数大于等于Threshold的关键词构建一个集合，用于作为共现矩阵的首行和首列
def get_set_key(dic, threshold):
    wf = {k: v for k, v in dic.items() if v >= threshold}
    set_key_list = []
    for a in sorted(wf.items(), key=lambda item: item[1], reverse=True):
        set_key_list.append(a[0])
    return set_key_list


'''建立矩阵，矩阵的高度和宽度为关键词集合的长度+1'''


def build_matirx(set_key_list):
    edge = len(set_key_list) + 1
    # matrix = np.zeros((edge, edge), dtype=str)
    matrix = [['' for j in range(edge)] for i in range(edge)]
    return matrix


'''初始化矩阵，将关键词集合赋值给第一列和第二列'''


def init_matrix(set_key_list, matrix):
    matrix[0][1:] = np.array(set_key_list)
    matrix = list(map(list, zip(*matrix)))
    matrix[0][1:] = np.array(set_key_list)
    return matrix


'''计算各个关键词共现次数'''


def count_matrix(matrix, commentwords_list):
    keywordlist = matrix[0][1:]  # 列出所有关键词
    appeardict = {}  # 每个关键词与 [出现在的行(formated_data)的list] 组成的dictionary
    for w in keywordlist:
        appearlist = []
        i = 0
        for each_line in commentwords_list:
            if w in each_line:
                appearlist.append(i)
            i += 1
        appeardict[w] = appearlist
    for row in range(1, len(matrix)):
        # 遍历矩阵第一行，跳过下标为0的元素
        for col in range(1, len(matrix)):
            # 遍历矩阵第一列，跳过下标为0的元素
            # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
            if col >= row:
                # 仅计算上半个矩阵
                if matrix[0][row] == matrix[col][0]:
                    # 如果取出的行关键词和取出的列关键词相同，则其对应的共现次数为0，即矩阵对角线为0
                    matrix[col][row] = str(0)
                else:
                    counter = len(set(appeardict[matrix[0][row]]) & set(appeardict[matrix[col][0]]))

                    matrix[col][row] = str(counter)
            else:
                matrix[col][row] = matrix[row][col]
    return matrix


def textrank(stopwords):
    """测试分词的效果"""
    textrank_keywords = []
    file = r'comment_list.txt'
    # 打开并读取文本文件
    text = codecs.open(file, 'r', 'utf-8').read()

    # 创建分词类的实例
    tr4w = TextRank4Keyword()
    # 对文本进行分析，设定窗口大小为2，并将英文单词小写
    tr4w.analyze(text=text, lower=True, window=2)

    """输出"""
    print('关键词为：')
    # 从关键词列表中获取前50个关键词
    for item in tr4w.get_keywords(num=50, word_min_len=1):
        # 去掉停用词
        if item.word not in stopwords:
            # 打印每个关键词的内容及关键词的权重
            print(item.word, item.weight)
            textrank_keywords.append(item.word)
    print('\n')
    return textrank_keywords


def main():
    # 异常处理
    try:
        # reload方法用于对已经加载的模块进行重新加载，一般用于原模块有变化的情况
        reload(sys)
        # 设置系统的默认编码方式，仅本次有效，因为setdefaultencoding函数在被系统调用后即被删除
        sys.setdefaultencoding('utf-8')
    except Exception as e:
        print(str(e))
        pass

    stopwords_path = 'stopwords.txt'
    # excel_path = 'raw_data.xlsx'
    output_path = r'co_matrix.txt'

    # 从数据库导入数据

    Data_admin = Database()
    output = Data_admin.select_data()
    # print(output)
    # print(type(output))
    comment_list_raw = []
    for i in range(len(output)):
        comment_list_raw.append(output[i][1])
    # 评论去重
    comment_list = list(set(comment_list_raw))
    # print(comment_list)

    # #从excel表格导入数据
    #
    # d = pd.read_excel(excel_path, sheetname=None)
    # # d['result'] = pd.DataFrame(['MachineLearning','DeepLearning'],index=['0','1'])
    # print(type((d['xiecheng'].content)))
    # comment_list_raw = d['xiecheng'].content.values
    # #评论去重
    # comment_list = list(set(comment_list_raw))
    # # print(d['xiecheng'].content)
    # #print(comment_list)
    #

    # 设置停用词
    print('start read stopwords data.')
    stopwords = []
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 0:
                stopwords.append(line.strip())

    file_path = "comment_list.txt"
    f = open(file_path, "w", encoding="utf-8")
    str_list = [line + '\n' for line in comment_list]  # 在list中加入换行符
    f.writelines(str_list)
    textrank_keywords = textrank(stopwords)

    key_list = []
    for i in range(len(comment_list)):
        words = tokenizer(comment_list[i], stopwords)
        key_list.extend(words)
    # 词频
    set_key_list = get_set_key(word_frequency_dict(key_list), 45)
    commentwords_list = format_data(comment_list, set_key_list, stopwords)

    # tfidf
    commentwords_list_duplication = format_data_duplication(comment_list, stopwords)
    result = getKeywords_tfidf(commentwords_list_duplication, 2)  # 每句评论的前五个关键词
    result.to_csv("keys_TFIDF.csv", index=False, encoding="utf_8_sig")

    # 同时考虑词频和tfidf和textrank得到关键词
    tfidf_list_raw = result['key'].drop_duplicates().values.tolist()
    tfidf_list = []
    for i in range(len(tfidf_list_raw)):
        tfidf_list.extend(tfidf_list_raw[i].split(' '))
    set_key_list_2 = list(set(set_key_list) & set(tfidf_list) & set(textrank_keywords))

    # 共现矩阵
    matrix = build_matirx(set_key_list_2)
    matrix = init_matrix(set_key_list_2, matrix)
    result_matrix = count_matrix(matrix, commentwords_list)
    np.savetxt(output_path, result_matrix, fmt=('%s,' * len(matrix))[:-1], encoding="utf-8")
    # print(result_matrix)


if __name__ == '__main__':
    main()
