# Tourism_NLP
## 软件要求
1. snownlp
2. pandas
3. jieba
4. 建议下载jupyter,通过清华镜像源安装会快一些  
命令
5. pymysql  
先更新一下pip  
`pip install --upgrade pip
`  
然后执行安装命令  
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyter`

然后终端输入`jupyter notebook
`查看能否正常使用  
基本使用参见博客  
https://blog.csdn.net/qq_33619378/article/details/83037106
## 自然语言处理入门
> Python课程材料

## jzg_wordcloud.ipynb
> 携程数据及词云分析/LDA

## sentiment1.0.py
> 情感分析示例。 

## 关于data_admin.py
请在本地数据库创建好tourism数据库，字符集为utf8mb4 -- UTF-8 Unicode，排序规则为utf8mb4_0900_ai_ci。需要用的数据表会发到群里，运行sql即可。  
在Database的类的初始函数中，请修改user和password属性为你自己的数据库用户和密码  
给database方法传入sql可执行相关操作，可接受的参数也可以自己研究一下