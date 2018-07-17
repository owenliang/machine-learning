# machine-learning

机器学习的重要示例, 偏工程应用.

## 说明

* lr_basic: 基于sklearn的LR分类
* lr_matlib: 将sklearn LR分类效果用matlib绘制出来
* lr_spark: 基于spark ml完成分布式训练, 模型保存到hdfs
* lr_pmml: 基于spark ml完成分布式训练, 模型导出pmml文件, 供跨语言加载(主要是JAVA)

## 依赖

* python3
* numpy
* scipy
* pandas: 用于dataframe处理
* matplotlib: 用于绘图
* sklearn: 单机机器学习算法
* pyspark: 分布式机器学习算法
* pyspark2pmml: 将spark ml的模型导出为pmml格式
* JPMML-SparkML: 支持spark ml模型转换pmml格式的jar包（[下载对应spark版本的JPMML-SparkML uber-JAR file](https://github.com/jpmml/pyspark2pmml)）

## 运行注意

### JPMML-SparkML jar

我为spark2.3下载的版本：jpmml-sparkml-executable-1.4.5.jar，放置在当前目录.

### 导出环境变量

export PYSPARK_PYTHON=/usr/local/bin/python3

这影响spark计算时用哪个版本的Python.