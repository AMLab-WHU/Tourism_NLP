# -*- coding: utf-8 -*-

import pandas as pd
from snownlp import SnowNLP
import jieba
from jieba import analyse


# 加载自定义词典
# jieba.load_userdict('C:/Users/cong/Desktop/网贷之家爬虫/平台.txt')

# 创建停用词list(平台名)
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词  
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


# 使用jieba获取关键词
def get_keywords(text):
    """
    # s = SnowNLP(text)
    https://www.jianshu.com/p/886880f19a09
    直接使用使用s = SnowNLP(text)效果不佳，分词准确度直接影响情感分析的准确度
    使用jieba进行关键词提取
    """
    line_seg = seg_sentence(text)
    # print(line_seg)
    kwds = " ".join(analyse.extract_tags(line_seg))
    # print(analyse.extract_tags(line_seg))
    # print(analyse.textrank(line_seg))
    '''
    https://blog.csdn.net/suibianshen2012/article/details/68927060
    实践对比了两种关键词提取方法，analyse.textrank(text) vs analyse.extract_tags(text)
    使用extract_tags基于TF-IDF算法进行关键词抽取效果更佳
    '''
    return kwds


# 使用snownlp获取文本情感值
def get_sentiment_cn(text):
    s = SnowNLP(text)
    return s.sentiments


if __name__ == '__main__':
    # 打开评论文件
    f = open('xiecheng.csv', encoding='utf-8')
    df = pd.read_csv(f)
    print(df.iloc[:, 0].size)
    # 计算df中评论的情感值
    df["keywords"] = df["content"].apply(get_keywords)
    df["sentiment"] = df["keywords"].apply(get_sentiment_cn)
    df.to_csv('xiecheng2.csv', index_label="index_label")
