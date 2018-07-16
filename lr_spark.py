# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
import pandas
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import  VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# 配置spark客户端
conf = SparkConf().setAppName("lr_spark")
conf = conf.setMaster("local")
sc = SparkContext(conf = conf)

# 加载sklearn的训练数据
iris = load_iris()
# 特征矩阵
features = pandas.DataFrame(iris.data, columns = iris.feature_names)
# 目标矩阵
targets = pandas.DataFrame(iris.target, columns = ['target'])
# 合并矩阵
merged = pandas.concat([features, targets], axis = 1)

# 创建SparkSession
sess = SparkSession(sc)

# 创建spark DataFrame
raw_df = sess.createDataFrame(merged)

# 把特征合并到features字段
vec_assem = VectorAssembler()
vec_assem.setInputCols(merged.columns.values[:-1]).setOutputCol('features')
train_df = vec_assem.transform(raw_df)

# 拆分训练集和测试集
train_df, test_df = train_df.randomSplit([0.8, 0.2])

# 创建LR分类器
lr = LogisticRegression(featuresCol = 'features', labelCol = 'target')

# 训练
train_df.show()
model = lr.fit(train_df)

# 预测test集合
predict_df = model.transform(test_df)


#predict_and_target = predict_df.rdd.map(lambda row: (float(model.predict(row.features)), row.target))
#print(predict_and_target.collect())

# 对测试集做predict, 生成(预测分类, 正确分类)
def build_predict_target(row):
    return (float(row.prediction), float(row.target))

predict_and_target_rdd = predict_df.rdd.map(build_predict_target)

# 统计模型效果
metrics = BinaryClassificationMetrics(predict_and_target_rdd)
print(metrics.areaUnderPR)

# 保存模型到磁盘
model.write().overwrite().save('./lr.model')