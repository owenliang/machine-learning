# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# GBDT算法
# from sklearn.ensemble import GradientBoostingClassifier

# 从官网下载数据
iris = load_iris()

# 随机拆分训练集与测试集
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size = 0.2)

# 逻辑回归分类算法
lr = LogisticRegression()

# 训练模型
lr.fit(train_x, train_y)

# 预测
predict_y = lr.predict(test_x)
print(predict_y)

# 模型得分
score = lr.score(test_x, test_y)
print(score)