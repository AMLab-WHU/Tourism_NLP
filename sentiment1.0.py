# -*- coding: utf-8 -*-

import pandas as pd
from snownlp import SnowNLP
import jieba
from jieba import analyse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

# 判断积极 or 消极 or 中性
def get_sentimentbyself(score):
    if score > 0.5:
        s_byself = "积极"
    elif score == 0.5:
        s_byself = "中性"
    else:
        s_byself = "消极"
    return s_byself


if __name__ == '__main__':
    # 打开评论文件
    f = open('xiecheng.csv', encoding='utf-8')
    df = pd.read_csv(f)
    print(df.iloc[:, 0].size)
    # 计算df中评论的情感值
    df["keywords"] = df["content"].apply(get_keywords)
    df["sentiment"] = df["content"].apply(get_sentiment_cn)
    # 改变评论时间的数据类型与格式
    df["time"] = df['time'].apply(lambda x: x.split(' ')[0])  # 时间只取日期，忽略具体时间点
    df["time"] = pd.to_datetime(df["time"], format='%Y-%m-%d')  # 将“time”列数据转换为时间类型
    #df.to_csv('xiecheng2.csv', index_label="index_label")
    #ceshi

    data = df[['time', 'sentiment']]  # 抽取出时间列和情感值列
    # data.sort_values('time', inplace=True)  # 根据时间列进行排序
    data.set_index('time', inplace=True)  # 将“time”列数据重置为索引，并扩展到前面的数据
    start_time = input("请输入您想要查看的时间段的开始日期（如 2015-12-31）：").replace("/n","")
    end_time = input("请输入您想要查看的时间段的结束日期（如 2015-12-31）：").replace("/n","")

    ## 绘制情感倾向人数图
    # 判断积极 or 消极 or 中性
    data["sentimentbyself"] = data["sentiment"].apply(get_sentimentbyself)
    #data.to_csv('xiecheng3.csv', index_label="index_label")
    data_p = data[(data.sentimentbyself == "积极")]
    data_n = data[(data.sentimentbyself == "消极")]
    data_z = data[(data.sentimentbyself == "中性")]
    data_p = data_p.resample('D').count()
    data_n = data_n.resample('D').count()
    data_z = data_z.resample('D').count()
    data_z.to_csv('xiecheng4.csv', index_label="index_label")
    plt.figure(figsize=(12, 8))
    plt.plot(data_p.loc[start_time:end_time].index, data_p.loc[start_time:end_time].sentimentbyself, label='Positive numbers')
    plt.plot(data_n.loc[start_time:end_time].index, data_n.loc[start_time:end_time].sentimentbyself, label='Negative numbers')
    plt.plot(data_z.loc[start_time:end_time].index, data_z.loc[start_time:end_time].sentimentbyself, label='Neutual numbers')
    plt.legend(loc='best')
    plt.title("Daily Trends")
    plt.show()

    ## 绘制每日情感值趋势图
    # 计算每日文本情感均值
    #data["sentiment"] = data.resample('D').mean()  # 按月采样，计算均值，问题是没有数据的行也会显示出来
    #data['sentiment'].fillna(0, inplace=True)  # 空值用0填充
    #data.to_csv('xiecheng3.csv', index_label="index_label")
    #data.loc[start_time:end_time].plot(figsize=(15, 8), title="Daily Sentiments", fontsize=8)
    # 配置横坐标
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    #plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    #plt.show()

