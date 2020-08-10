# -*- coding: utf-8 -*-

from data_admin import Database
from snownlp import SnowNLP

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

## 执行程序
if __name__ == '__main__':
    insert_score()