---
title: lightGBM用法详解
mathjax: true
toc: true
comments: true
date: 2019-10-30 14:29:50
categories:
tags:
---



<!--more-->

# leaves_wise

# 参数说明

- objective: 任务类型
    1. 对于回归问题，可以指定为：regression_l2, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie
    2. 对于分类问题，可以指定为：binary, multiclass, multiclassova
- featuresCol：特征列名称
- labelCol：标签列名称
- alpha：huber、quantile用到的参数
- tweedieVariancePower：控制tweedie分布的参数，大小介于1到2之间。该值等于1时，表示的是泊松分布；等于2时，表示的是伽马分布
- boostingType：模型类型，可以指定为：gbdt, gbrt, rf(Random Forest), random_forest, dart(Dropouts meet Multiple Additive Regression Trees), goss(Gradient-based One-Side Sampling)
- featureFraction：特征采样比例（0~1)。每棵树的特征子集占比，大小介于0到1之间
- baggingFraction：样本采样比例(0~1)。不进行重采样的随机选取部分样本数据
- baggingFreq：样本采样频率，指定每几轮迭代重新进行一次采样(必须和baggingFraction同时设置)
- baggingSeed：设置随机采样的种子值
- boostFromAverage：true表示初始的分数使用标签的均值，加速模型训练的收敛速度。仅用于回归任务
- categoricalSlotIndexes：指定离散特征在特征列中的下标
- categoricalSlotNames：指定离散特征在特征列中的名称
- defaultListenPort：默认的执行器监听端口，测试中使用
- earlyStoppingRound：当验证集效果不发生提升的最大迭代窗口超过指定值，就提前终止模型训练。0表示不进行训练的提前终止
- initScoreCol：指定用于初始化分数的列名称，用于继续训练
- isProvideTrainingMetric：指定是否输出训练数据的模型评估结果
- lambdaL1：l1
- lambdaL2：l2
- learningRate：学习率/收缩率
- maxBin：
- maxDepth：最大树深度，0表示只有一个叶子节点
- minSumHessianInLeaf：
- modelString：模型文件地址
- numBatches：批次大小，0表示不分批训练
- numIterations：迭代次数
- numLeaves：叶子节点数量
- parallelism：
    - data_parallel
    - voting_parallel
- predictionCol：预测结果列名称
- timeout：
- useBarrierExecutionMode：
- validationIndicatorCol：
- verbosity：程序输出日志级别
    - <0：Fatal
    - 0：Error
    - 1：Info
    - >1：Debug
- weightCol：
- probabilityCol：
- rawPredictionCol：
- thresholds：多分类任务中用于调整预测每个类的概率
    - 数组的长度必须和类别个数相同
    - 必须满足t.forall(_ >= 0) && t.count(_ == 0) <= 1
    - P_adjust(c) = P_raw(c) / threshold(c)
- generateMissingLabels：
- isUnbalance：用于在二分类中，说明训练数据是否平衡
