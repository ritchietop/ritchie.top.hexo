---
title: 【Tensorflow2】Estimator使用流程——以Titanic预测任务为例
mathjax: false
toc: false
comments: true
date: 2020-02-25 12:01:20
categories: tensorflow
tags: estimator titanic
---

> Estimator是Tensorflow完整模型的高级表示，它被设计用于轻松扩展和异步训练。

这句话是官方文档上的描述，我觉得说的挺对的。为了更直观的体会这句话，这里我以Kaggle经典的入门练习Titanic为例，为大家呈现Estimator的完整使用流程。

<!--more-->

# Titanic任务与数据说明

当完成Kaggle的注册之后，你需要完成的第一个入门级任务就是[Titanic](https://www.kaggle.com/c/titanic)。
这个任务的目的是预测乘客是否会存活。数据集中会提供乘客的一些基础特征，用于构建模型输入数据。这些特征包括了：
1. PassengerId: 乘客的ID
2. Survived: 乘客是否存活。0表示没有存活，1表示存活
3. Pclass: 乘客的船票等级。包含1、2、3级。
4. Name: 乘客姓名。
5. Sex: 乘客性别。包括male和female。
6. Age: 乘客年龄。
7. SibSp: 和乘客一起在船上的亲人数（包括兄弟姐妹配偶）。
8. Parch: 和乘客一起在船上的亲人数（包括父母和子女）。
9. Ticket: 乘客的船票编号。
10. Fare: 乘客的船票价格。
11. Cabin: 乘客的船舱编号。
12. Embarked: 乘客的上船港口。包括C(瑟堡)、Q(皇后镇)、S(南安普顿)

上面的PassengerId作为索引列，不参与到模型特征构建。Survived为模型预测的标签列。很明显这是个二分类问题。话不多说，接下来我将对数据进行简单的预处理，以方便后面的模型训练。

# 数据预处理

由于我后面会使用FeatureColumn来对特征进行编码，这里的预处理只是进行简单的缺失值填充。通过对每个特征列取值的统计，发现`Age`、`Fare`、`Cabin`、`Embarked`这四个特征列中存在取值缺失的情况。这里我直接采用给缺失记录统一填充相同缺失标志符的方法来处理。

```python
import pandas as pd

def data_process(data):
    data['Age'] = pd.to_numeric(data['Age'].fillna(-1), errors='raise')
    data['Fare'] = pd.to_numeric(data['Fare'].fillna(-1), errors='raise')
    data['Cabin'] = data['Cabin'].fillna('nan')
    data['Embarked'] = data['Embarked'].fillna('nan')
    return data
```

缺失值处理的方法有很多，这里我只是为了呈现Estimator的完整使用流程，对于模型的准确率并不是本篇文章的关注点。

# input_fn

这里用来构建模型训练和评估时的输入函数。由于是包含了训练和评估两部分，就需要对数据集进行分隔。
Titanic的数据集下载地址在[这里](https://www.kaggle.com/c/titanic/download/GQf0y8ebHO0C4JXscPPp%2Fversions%2FXkNkvXwqPPVG0Qt3MtQT%2Ffiles%2Ftrain.csv)
我会按照8:2的比例来获得训练数据集和测试数据集。

```python
import pandas as pd

def split_data(path, test_rate):
    data = pd.read_csv(filepath_or_buffer=path)
    data = data_process(data)
    test_data = data.sample(frac=test_rate)
    train_data = pd.concat([data, test_data], axis=0)
    train_data.drop_duplicates(keep=False, inplace=True)
    return train_data, test_data

train_data, test_data = split('./train.csv', test_rate=0.2)
```

上面这种划分还是比较傻的，sklearn中提供了很多更好的方法，有兴趣的朋友可以去研究一下。

接下来，就可以定义模型的输入函数了。我将根据具体的代码实现来说明一下构建逻辑：

```python
def input_fn(data, batch_size, is_training, num_epochs=1):
    labels = data.pop('Survived')
    dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.repeat(count=num_epochs)
    return dataset.batch(batch_size=batch_size)
```

这里的输入包括了上面切割好的训练数据和测试数据，就是参数data。
还需要定义数据的批次大小(batch_size)，Estimator采用小批量梯度下降的方法来进行模型训练。
is_training用来区分任务类型，以此来决定是否需要对数据进行shuffle操作。训练过程中对数据进行shuffle，可以在一定程度上加快模型收敛速度。
训练任务一般会对数据进行多轮迭代，通过num_epochs可以设置迭代次数。

这一部分主要使用tf.data的API，更多内容可以参考[Dataset简明教程](https://www.ritchie.top/2019/12/28/tensorflow-dataset/)

# feature_column

原始的输入数据中既有连续特征也有离散特征，这就需要我给每个特征的定义处理逻辑，来完成原始数据向模型真正用于计算的输入数据的数值化转换。
我这里并不打算使用所有的特征列，只用其中的几个进行举例即可。

- 可以穷举的标称特征

Sex、Pclass、Embarked这三个特征的取值不多，可以使用list进行可能取值的管理。
```python
sex = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Sex', vocabulary_list=['male', 'female'])
pclass = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Pclass', vocabulary_list=[1, 2, 3])
embarked = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Embarked', vocabulary_list=['S', 'C', 'Q'])
```

- 无法穷举的标称特征

Ticket、Cabin这两个特征的取值比较多，无法使用list来进行可能取值的管理。
```python
ticket = tf.feature_column.categorical_column_with_hash_bucket(
    key='Ticket', hash_bucket_size=1000)
cabin = tf.feature_column.categorical_column_with_hash_bucket(
    key='Cabin', hash_bucket_size=300)
```

- 连续特征

SibSp、Parch都是数值类型，可以直接用到模型中。
```python
sibsp = tf.feature_column.numeric_column(key='SibSp')
parch = tf.feature_column.numeric_column(key='Parch')
```

- 数值类型特征的离散化处理

Age、Fare也是数值类型，但是Age特征不具有计算意义，Fare特征的取值差异性较大。针对这两个特征我进行分桶处理。
```python
raw_age = tf.feature_column.numeric_column(key='Age')
age = tf.feature_column.bucketized_column(
    source_column=raw_age, boundaries=[0, 1, 10, 18, 30, 40, 50, 60, 70])
raw_fare = tf.feature_column.numeric_column(key='Fare')
fare = tf.feature_column.bucketized_column(
    source_column=raw_fare, boundaries=[0, 8, 12, 20, 30, 60, 120])
```

这里我暂时只使用上面这些特征，完整的处理逻辑如下：

```python
def feature_column():
    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Sex', vocabulary_list=['male', 'female'])
    pclass = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Pclass', vocabulary_list=[1, 2, 3])
    embarked = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Embarked', vocabulary_list=['S', 'C', 'Q'])
    ticket = tf.feature_column.categorical_column_with_hash_bucket(
        key='Ticket', hash_bucket_size=1000)
    cabin = tf.feature_column.categorical_column_with_hash_bucket(
        key='Cabin', hash_bucket_size=300)
    sibsp = tf.feature_column.numeric_column(key='SibSp')
    parch = tf.feature_column.numeric_column(key='Parch')
    raw_age = tf.feature_column.numeric_column(key='Age')
    age = tf.feature_column.bucketized_column(
        source_column=raw_age, boundaries=[0, 1, 10, 18, 30, 40, 50, 60, 70])
    raw_fare = tf.feature_column.numeric_column(key='Fare')
    fare = tf.feature_column.bucketized_column(
        source_column=raw_fare, boundaries=[0, 8, 12, 20, 30, 60, 120])

    return [
        tf.feature_column.embedding_column(categorical_column=sex, dimension=2),
        tf.feature_column.embedding_column(categorical_column=pclass, dimension=2),
        tf.feature_column.embedding_column(categorical_column=embarked, dimension=2),
        tf.feature_column.embedding_column(categorical_column=ticket, dimension=10),
        tf.feature_column.embedding_column(categorical_column=cabin, dimension=8),
        sibsp, parch,
        tf.feature_column.embedding_column(categorical_column=age, dimension=3),
        tf.feature_column.embedding_column(categorical_column=fare, dimension=3)
    ]
```

在最后的返回特征列表中，我对所有的离散化特征列进行了Embedding处理，从而将最终的所有特征列都进行了数值化转换。

# Estimator模型

Tensorflow官方提供了一些开箱即用的Estimator模型，这里我使用其中的DNNClassifier模型来进行此次任务的预测。关于如何实现自定义Estimator模型，后续我还会再分享。

```python
model = tf.estimator.DNNClassifier(feature_columns=feature_column(),
                                   activation_fn='relu',
                                   batch_norm=True,
                                   dropout=0.7,
                                   hidden_units=[256, 128, 64],
                                   optimizer='Adam',
                                   model_dir='./model',
                                   n_classes=2)
```

这样就定义好了一个DNN二分类模型，里面的很多参数都可以调整。关于模型调优的这里就不说了。

# 训练与评估

Estimator设计了一套非常方便的模型训练和评估架构，能够实现一套模型代码无缝地在单机和分布式环境下任意切换。人性化的方法设计也让模型训练和评估变得更加直观简单。

```python
train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(train_input, 100, True, 50), max_steps=None)
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: input_fn(test_input, 100, False, 1), steps=None)
```

通过定义TrainSpec和EvalSpec来配置模型训练和评估的参数。这里主要用到的就是input_fn和最大训练或评估步数。

触发模型的训练和评估，使用下面的方法实现：

```python
tf.estimator.train_and_evaluate(
    estimator=model, train_spec=train_spec, eval_spec=eval_spec)
```

使用到的参数都是上面已经定义好的，只需要把它们捆绑到一起即可。

# 完整代码和日志

```python
import tensorflow as tf
import pandas as pd

def data_process(data):
    data['Age'] = pd.to_numeric(data['Age'].fillna(-1), errors='raise')
    data['Fare'] = pd.to_numeric(data['Fare'].fillna(-1), errors='raise')
    data['Cabin'] = data['Cabin'].fillna('nan')
    data['Embarked'] = data['Embarked'].fillna('nan')
    return data

def split_data(path, test_rate):
    data = pd.read_csv(filepath_or_buffer=path)
    data = data_process(data)
    test_data = data.sample(frac=test_rate)
    train_data = pd.concat([data, test_data], axis=0)
    train_data.drop_duplicates(keep=False, inplace=True)
    return train_data, test_data

def input_fn(data, batch_size, is_training, num_epochs=1):
    labels = data.pop('Survived')
    dataset = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.repeat(count=num_epochs)
    return dataset.batch(batch_size=batch_size)

def feature_column():
    sex = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Sex', vocabulary_list=['male', 'female'])
    pclass = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Pclass', vocabulary_list=[1, 2, 3])
    embarked = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Embarked', vocabulary_list=['S', 'C', 'Q'])
    ticket = tf.feature_column.categorical_column_with_hash_bucket(
        key='Ticket', hash_bucket_size=1000)
    cabin = tf.feature_column.categorical_column_with_hash_bucket(
        key='Cabin', hash_bucket_size=300)
    sibsp = tf.feature_column.numeric_column(key='SibSp')
    parch = tf.feature_column.numeric_column(key='Parch')
    raw_age = tf.feature_column.numeric_column(key='Age')
    age = tf.feature_column.bucketized_column(
        source_column=raw_age, boundaries=[0, 1, 10, 18, 30, 40, 50, 60, 70])
    raw_fare = tf.feature_column.numeric_column(key='Fare')
    fare = tf.feature_column.bucketized_column(
        source_column=raw_fare, boundaries=[0, 8, 12, 20, 30, 60, 120])

    return [
        tf.feature_column.embedding_column(categorical_column=sex, dimension=2),
        tf.feature_column.embedding_column(categorical_column=pclass, dimension=2),
        tf.feature_column.embedding_column(categorical_column=embarked, dimension=2),
        tf.feature_column.embedding_column(categorical_column=ticket, dimension=10),
        tf.feature_column.embedding_column(categorical_column=cabin, dimension=8),
        sibsp, parch,
        tf.feature_column.embedding_column(categorical_column=age, dimension=3),
        tf.feature_column.embedding_column(categorical_column=fare, dimension=3)
    ]

model = tf.estimator.DNNClassifier(feature_columns=feature_column(),
                                   activation_fn='relu',
                                   batch_norm=True,
                                   dropout=0.7,
                                   hidden_units=[256, 128, 64],
                                   optimizer='Adam',
                                   model_dir='./model',
                                   n_classes=2)

train_input, test_input = split_data('./train.csv', test_rate=0.2)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(train_input, 100, True, 50), max_steps=None)
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: input_fn(test_input, 100, False, 1), steps=None)

tf.estimator.train_and_evaluate(
    estimator=model, train_spec=train_spec, eval_spec=eval_spec)
```

```text
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_model_dir': './model', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x145cef290>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Not using Distribute Coordinator.
INFO:tensorflow:Running training and evaluation locally (non-distributed).
INFO:tensorflow:Start train and evaluate loop. The evaluate will happen after every checkpoint. Checkpoint frequency is determined based on RunConfig arguments: save_checkpoints_steps None or save_checkpoints_secs 600.
INFO:tensorflow:Calling model_fn.
WARNING:tensorflow:Layer dnn is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into ./model/model.ckpt.
INFO:tensorflow:loss = 0.9460948, step = 0
INFO:tensorflow:global_step/sec: 108.812
INFO:tensorflow:loss = 0.5249347, step = 100 (0.920 sec)
INFO:tensorflow:global_step/sec: 215.037
INFO:tensorflow:loss = 0.4911957, step = 200 (0.465 sec)
INFO:tensorflow:global_step/sec: 227.797
INFO:tensorflow:loss = 0.48654288, step = 300 (0.439 sec)
INFO:tensorflow:Saving checkpoints for 357 into ./model/model.ckpt.
INFO:tensorflow:Calling model_fn.
WARNING:tensorflow:Layer dnn is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.

If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.

To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.

INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2020-02-25T13:25:28Z
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./model/model.ckpt-357
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2020-02-25-13:25:29
INFO:tensorflow:Saving dict for global step 357: accuracy = 0.7303371, accuracy_baseline = 0.6348315, auc = 0.8478557, auc_precision_recall = 0.7938756, average_loss = 0.5330257, global_step = 357, label/mean = 0.36516854, loss = 0.53750694, precision = 0.9047619, prediction/mean = 0.24691269, recall = 0.2923077
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 357: ./model/model.ckpt-357
INFO:tensorflow:Loss for final step: 0.46987808.
```

# 模型预测

完成了上面的操作，就可以在`./model`目录下得到最终的模型文件。关于模型文件的格式、导出等更多内容，我后续也会力争给大家分享。

Titanic的预测数据集下载地址在[这里](https://www.kaggle.com/c/titanic/download/GQf0y8ebHO0C4JXscPPp%2Fversions%2FXkNkvXwqPPVG0Qt3MtQT%2Ffiles%2Ftest.csv)

由于预测数据集并不包含`Survived`列，所以我重新写了个input_fn方法，只包含了批次设置。
预测的时候，直接使用上面定义的`model`变量，调用predict方法就可以了，不能再简单。

```python
predict_input = data_process(pd.read_csv('./test.csv'))

def predict_input_fn(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dict(data))
    dataset = dataset.batch(batch_size=batch_size)
    return dataset

predictions = model.predict(input_fn=lambda: predict_input_fn(predict_input, 100))
for prediction, passengerId in zip(predictions, predict_input['PassengerId']):
    classes = prediction['classes'][0]
    print(passengerId, int(classes))
```
