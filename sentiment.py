# -*- coding: utf-8 -*-

from data_admin import Database
from snownlp import SnowNLP
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## 使用snownlp获取文本情感值
def get_sentiment_cn(text):
    s = SnowNLP(text)
    return s.sentiments

## 判断情感属性：积极 or 消极 or 中性
def get_sentimentbyself(score):
    if score > 0.5:
        s_byself = "positive"
    elif score == 0.5:
        s_byself = "neutral"
    else:
        s_byself = "negative"
    return s_byself

## 插入情感值和情感倾向
def insert_score():
    Data_admin = Database()
    sql_1 = 'select autoId, content from sentiment'
    data_1 = Data_admin.database(sql_1)
    sql_2 = 'select autoId, content from ctrip_comments'
    data_2 = Data_admin.database(sql_2)
    data_list=[]
    for text in data_2:
        score = get_sentiment_cn(text[1])
        s_byself = get_sentimentbyself(score)
        if text in data_1:
            sql_3 = 'update sentiment set Score="{}", Sentimentbyself="{}" where autoId={}'.format(score, s_byself, text[0])
            Data_admin.database(sql_3)
        else:
            data_list.append((text[0], text[1], score, s_byself))
    sql = 'insert into sentiment(autoId,content,Score,Sentimentbyself) values (%s, %s, %s,%s)'
    Data_admin.database(sql,data_list=data_list)

## 时间数据处理：获取两时间点内的日期列表
def get_date_list():
    date_list = []
    start = input("请输入您想要查看的时间段的开始日期（如 2015-12-31）：").replace("/n", "")
    end = input("请输入您想要查看的时间段的结束日期（如 2015-12-31）：").replace("/n", "")
    date_start = datetime.datetime.strptime(start, '%Y-%m-%d')
    date_end = datetime.datetime.strptime(end, '%Y-%m-%d')
    while date_start <= date_end:
        date_list.append(date_start.strftime('%Y-%m-%d'))
        date_start += datetime.timedelta(days=1)
    return date_list

## 对字典进行处理
def get_data_list(dict_data):
    dict_data = sorted(dict_data.items(), key=lambda x: x[0])  # 升序排序时间
    final_date_list = []  # 存储排序过的时间
    data_list = [] # 存储对应的情感倾向人数列表
    for m in dict_data:
        final_date_list.append(m[0])
        data_list.append(m[1])
    return final_date_list,data_list

## 绘制情感倾向人数图
def draw_trend():
    date_list = get_date_list()
    dict_data_p = {} # 存储时间与积极倾向评论数的对应关系
    dict_data_n = {}  # 存储时间与消极倾向评论数的对应关系
    dict_data_z = {}  # 存储时间与中性倾向评论数的对应关系
    Data_admin = Database()
    sql = 'select publishTime,Sentimentbyself from ctrip_comments, sentiment where ctrip_comments.autoId=sentiment.autoId'
    data = Data_admin.database(sql)
    for term in date_list:
        p, n, z = 0, 0, 0
        for text in data:
            if text[0].strftime('%Y-%m-%d') == term:
                if text[1] == "positive":
                    p += 1
                elif text[1] == "negative":
                    n += 1
                elif text[1] == "neutral":
                    z += 1
        dict_data_p[term] = p
        dict_data_n[term] = n
        dict_data_z[term] = z
    final_date_p, data_list_p = get_data_list(dict_data_p)
    final_date_n, data_list_n = get_data_list(dict_data_n)
    final_date_z, data_list_z = get_data_list(dict_data_z)
    if final_date_p == final_date_n == final_date_z :
        plt.figure(figsize=(10, 6))
        plt.plot(final_date_p, data_list_p, marker='o',label='Positive numbers')
        plt.plot(final_date_n, data_list_n, marker='o',label='Negative numbers')
        plt.plot(final_date_z, data_list_z, marker='o',label='Neutual numbers')
        plt.title("Daily Trends")
        plt.xticks(rotation=90, size = 8) #横坐标旋转90度
        plt.legend(loc='best')  # 显示上面的label
        plt.show()




## 执行程序
if __name__ == '__main__':
    #insert_score()
    draw_trend()