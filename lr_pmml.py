# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
import pandas
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark2pmml import PMMLBuilder
from pyspark.ml.feature import RFormula

# 配置spark客户端
conf = SparkConf().setAppName("lr_spark").set("spark.jars", "./jpmml-sparkml-executable-1.4.5.jar") # 注意: 这里需要加载jpmml jar
conf = conf.setMaster("local")
sc = SparkContext(conf = conf)

# 加载sklearn的训练数据
iris = load_iris()
# 特征矩阵
features = pandas.DataFrame(iris.data, columns = iris.feature_names)
# 目标矩阵
targets = pandas.DataFrame(iris.target, columns = ['Species'])
# 合并矩阵
merged = pandas.concat([features, targets], axis = 1)

# 创建SparkSession
sess = SparkSession(sc)

# 创建spark DataFrame
raw_df = sess.createDataFrame(merged)

# 特征提取
fomula = RFormula(formula = 'Species ~ .')

# 创建LR分类器
lr = LogisticRegression()

# 流水线: 先提取特征, 再训练模型
pipeline = Pipeline(stages = [fomula, lr])
pipeline_model = pipeline.fit(raw_df)

# 导出PMML
pmmlBuilder = PMMLBuilder(sc, raw_df, pipeline_model)
pmmlBuilder.buildFile("lr.pmml")