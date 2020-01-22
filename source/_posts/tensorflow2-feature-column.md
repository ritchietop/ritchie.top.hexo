---
title: 【Tensorflow2】FeatureColumn简明教程
mathjax: true
toc: false
comments: true
date: 2019-12-28 17:55:28
categories: tensorflow
tags: feature_column embedding_column FeatureTransformationCache StateManager
---

`tf.feature_column`是官方提供的一套用于处理结构化数据的工具。它是原始数据和`Estimator`模型之间的桥梁。丰富的`tf.feature_column`方法，让你可以将各种原始数据转换为`Estimators`可以使用的格式，从而更加容易的进行模型实验。

<!--more-->

{% asset_img inputs_to_model_bridge.jpg 数据与模型之间的桥梁 %}

这里我将对`tf.feature_column`目前支持的16种方法的使用进行说明和演示。通过下面的图示可以看到，`tf.feature_column`支持的所有特征列都继承自`FeatureColumn`类。通过`FeatureColumn`又引申出三个子类`CategoricalColumn`、`DenseColumn`、`SequenceDenseColumn`分别对应离散特征、连续特征、连续序列特征。如果我们想自定义类似`tf.feature_column`的方法，就需要通过继承这三个特征列来覆写对应的方法。不过这里只介绍官方的方法，自定义部分后续会再分享。

{% asset_img feature_column_methods.jpg 数据与模型之间的桥梁 %}

除了上面`tf.feature_column`方法依赖关系的介绍，为了方便后面的演示，还需要了解两个很重要的类`FeatureTransformationCache`和`StateManager`。
`FeatureTransformationCache`是输入数据的持有工具，它能够将输入进行缓存，从而方便后续对特征的多次使用。`FeatureTransformationCache`的本质就是`dict`。
`StateManager`是为具有状态数据的特征列提供状态管理，主要涉及状态的创建、增加、获取等。官方给出的一种实现是将`StateManager`与`Layer`关联，将状态数据存到`Layer`中进行管理。
预备知识基本就这些，下面我们来对每个`tf.feature_column`方法依次进行说明。

# CategoricalColumn系列

`CategoricalColumn`派生出的10个特征处理方法功能上是类似的，区别主要在于离散特征映射为数值类型的方法不同：

- `tf.feature_column.categorical_column_with_vocabulary_list`：通过定义离散特征的取值列表，将离散特征映射为其对应的列表下标（从0开始）。
- `tf.feature_column.sequence_categorical_column_with_vocabulary_list`：同上，区别是输入数据是离散序列特征。
- `tf.feature_column.categorical_column_with_vocabulary_file`：通过定义离散特征的取值文件，将离散特征映射为其对应的文件行数（从0开始）。
- `tf.feature_column.sequence_categorical_column_with_vocabulary_file`：同上，区别是输入数据是离散序列特征。
- `tf.feature_column.categorical_column_with_identity`：是将数值特征视为离散特征，并直接映射到自身。
- `tf.feature_column.sequence_categorical_column_with_identity`：同上，区别是输入数据是离散序列特征。
- `tf.feature_column.categorical_column_with_hash_bucket`：通过定义hash的空间大小，将离散特征映射为其对应的哈希值。
- `tf.feature_column.sequence_categorical_column_with_hash_bucket`：同上，区别是输入数据是离散序列特征。
- `tf.feature_column.crossed_column`：通过定义hash的空间大小，对指定特征列表中的所有列进行交叉，将交叉后的离散值映射为其对应的哈希值。
- `tf.feature_column.weighted_categorical_column`：这个是带权特征列，只是将`CategoricalColumn`和`DenseColumn`进行组合，本身并不定义映射方法。

## CategoricalColumn自定义方法

### get_sparse_tensors(transformation_cache, state_manager)

该方法会返回一个IdWeightPair(id_tensor, weight_tensor)。id_tensor表示离散特征进过数值映射（例如hash）之后的SparseTensor形式，其对应的values是离散特征映射后的数值（注意这里并不是one-hot/multi-hot的SparseTensor形式）。weight_tensor表示离散特征对应的权重值。

### num_buckets()

该方法会返回一个数值，表示离散特征映射为数值类型后的取值空间大小，也就是通过one-hot编码后的向量维度大小。

## categorical_column_with_vocabulary_list

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def categorical_list_column():
    column = tf.feature_column.categorical_column_with_vocabulary_list(
        # 特征列的名称
        key="feature",
        # 有效取值列表，列表的下标对应转换的数值。即，value1会被映射为0
        vocabulary_list=["value1", "value2", "value3"],
        # 取值的类型，只支持string和integer，这个会根据vocabulary_list自动推断出来
        dtype=tf.string,
        # 当取值不在vocabulary_list中时，会被映射的数值，默认为-1
        # 当该值不为-1时，num_oov_buckets必须设置为0。即两者不能同时起作用
        default_value=-1,
        # 作用同default_value，但是两者不能同时起作用。
        # 将超出的取值映射到[len(vocabulary), len(vocabulary) + num_oov_buckets)内
        # 默认取值为0
        # 当该值不为0时，default_value必须设置为-1
        # 当default_value和num_oov_buckets都取默认值时，会被映射为-1
        num_oov_buckets=3)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值可以为Tensor，也可以为SparseTensor
        "feature": tf.constant(value=[
            [["value1", "value2"], ["value3", "value3"]],
            [["value3", "value5"], ["value4", "value4"]]
        ])
    })
    # IdWeightPair(id_tensor, weight_tensor)
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)

def sequence_categorical_list_column():
    # 用法同categorical_column_with_vocabulary_list完全一致
    column = tf.feature_column.sequence_categorical_column_with_vocabulary_list(
        key="feature",
        vocabulary_list=["value1", "value2", "value3"],
        dtype=tf.string,
        default_value=-1,
        num_oov_buckets=2)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            ["value1", "value2", "value3", "value3"],
            ["value3", "value5", "value4", "value4"]
        ])
    })
    # IdWeightPair(id_tensor, weight_tensor)
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
```

## categorical_column_with_vocabulary_file

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def categorical_file_column():
    # 用法同categorical_column_with_vocabulary_list
    # 区别在于这个使用文件来管理取值集合，每一行代表一种取值，使用行号作为映射值
    column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="feature",
        vocabulary_file="path/valuelist",
        dtype=tf.string,
        default_value=-1,
        num_oov_buckets=3)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值可以为Tensor，也可以为SparseTensor
        "feature": tf.SparseTensor(
            indices=[
                [0, 0, 2],
                [0, 0, 3],
                [0, 2, 1],
                [1, 0, 1],
                [1, 1, 3]
            ],
            values=["value1", "value2", "value3", "value4", "value1"],
            dense_shape=[2, 3, 5]
        )
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)

def sequence_categorical_file_column():
    # 用法同categorical_column_with_vocabulary_file完全一致
    column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
        key="feature",
        vocabulary_file="path/valuelist",
        dtype=tf.string,
        default_value=-1,
        num_oov_buckets=3)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [["value1", "value2"], ["value3", "value3"]],
            [["value3", "value5"], ["value4", "value4"]]
        ])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
```

## categorical_column_with_identity

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def categorical_identity_column():
    # 如果特征类型为integer，并且取值在[0, num_buckets)之间
    # 等价于categorical_column_with_vocabulary_list中的list为[0 ~ num_bucket)
    column = tf.feature_column.categorical_column_with_identity(
        key='feature',
        # 取值范围为[0, num_buckets)
        num_buckets=10,
        # 数据不在[0, num_buckets)内时，将被映射的值。
        # 默认为None，这种情况下，当存在未知数据，会报错。
        # 要求default_value的取值在[0, num_buckets)内
        default_value=3)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值可以为Tensor，也可以为SparseTensor
        "feature": tf.constant(value=[
            [1, 2, 3, 4, 5, 6],
            [5, 6, 7, 8, 9, 10],
            [8, 9, 10, 11, 12, 13]
        ])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)

def sequence_categorical_identity_column():
    # 用法同categorical_column_with_identity完全一致
    column = tf.feature_column.sequence_categorical_column_with_identity(
        key='feature',
        num_buckets=10,
        default_value=3)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [[1, 2, 3], [4, 5, 6]],
            [[5, 6, 7], [8, 9, 10]],
            [[8, 9, 10], [11, 12, 13]]
        ])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
```

## categorical_column_with_hash_bucket

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def categorical_hash_column():
    # 如果特征取值比较多，无法全部罗列出来，可以使用hash映射的方法
    # 这里会将取值经过hash之后，映射到[0, hash_bucket_size)内
    column = tf.feature_column.categorical_column_with_hash_bucket(
        key="feature",
        # hash的空间大小
        hash_bucket_size=5000,
        # 只支持string和integer
        # 数值类型也是进行hash映射
        dtype=tf.string)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值可以为Tensor，也可以为SparseTensor
        "feature": tf.constant(value=[
            [[["value1"], ["value2"]], [["value3"], ["value3"]]],
            [[["value3"], ["value5"]], [["value4"], ["value4"]]]
        ])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)

def sequence_categorical_hash_column():
    # 用法同categorical_column_with_hash_bucket完全一致
    column = tf.feature_column.sequence_categorical_column_with_hash_bucket(
        key="feature",
        hash_bucket_size=5000,
        dtype=tf.string)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [[["value1"], ["value2"]], [["value3"], ["value3"]]],
            [[["value3"], ["value5"]], [["value4"], ["value4"]]]
        ])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
```

## crossed_column

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def cross_column():
    column = tf.feature_column.crossed_column(
        # 只支持2-D特征（包含batch维度）
        # key的类型可以为string或者CategoricalColumn（不支持HashCategoricalColumn）
        keys=["feature1", "feature2"],
        hash_bucket_size=1000,
        # FingerprintCat64对特征进行hash时使用，这个使用默认值就可以了
        # integer类型
        hash_key=None)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值可以为Tensor，也可以为SparseTensor
        # 只对有值的数据进行交叉，下面的例子，两条记录的交叉个数分别为12和9
        "feature1": tf.constant(value=[
            ["value11", "value12", "value13"],
            ["value11", "value11", "value14"]
        ]),
        "feature2": tf.SparseTensor(
            # indices要按顺序写
            indices=[
                [0, 1],
                [0, 3],
                [0, 5],
                [0, 6],
                [1, 0],
                [1, 2],
                [1, 4]
            ], 
            values=[4, 1, 7, 9, 3, 4., 4], 
            dense_shape=[2, 7])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
```

## weighted_categorical_column

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def weighted_column():
    sub_categorical_column = tf.feature_column.categorical_column_with_identity(
        key="feature",
        num_buckets=5,
        default_value=0)
    # 没有key和weight长度的匹配判断。所以，两者最终输出的shape可能不一致
    # 最终返回的key和weight都是SparseTensor
    column = tf.feature_column.weighted_categorical_column(
        # 这个也可以使用sequence列，但是两者返回的结果是一致的
        categorical_column=sub_categorical_column,
        weight_feature_key="feature_weight",
        # 类型只能为integer或float
        # 需要保持和weight_feature_key特征的匹配
        # 不过最终返回的weight_tensor也会被强制转成float32
        dtype=tf.float32)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # weighted column只是将两列合并为一列，并没有对两列的关系做具体限制
        # 所以具体key和value的输入要求与原始column一致
        "feature": tf.constant(value=[
            [[[1, 2]], [[3, 4]], [[5, 5]]],
            [[[9, 8]], [[7, 6]], [[5, 4]]]
        ]),
        "feature_weight": tf.constant(value=[
            [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
            [[9.9, 8.8, 7.7, 6.6, 5.5, 4.4]]
        ])
    })
    return column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
```

# DenseColumn系列

`DenseColumn`派生出的方法只有`numeric_column`，就是简单的获得输入的连续特征数据。

## DenseColumn自定义方法

### get_dense_tensor(transformation_cache, state_manager)

这个方法返回一个Tensor，表示的就是最终输出的连续特征数据。

### variable_shape()

这个方法返回一个TensorShape，表示`get_dense_tensor`返回值的shape大小（不包含batch维度）。

## numeric_column

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def numeric_column():
    # shape, default_value, dtype一起定义了特征列解析规则
    # tf.io.FixedLenFeature(shape, default_value, dtype)
    # 如果加载数据时，自己指定特征解析规则，这几个参数就没啥用
    column = tf.feature_column.numeric_column(
        key="feature",
        shape=(5,),
        default_value=0,
        # 类型为integer或者float
        dtype=tf.float32,
        # 数值都会执行该方法转化之后再返回
        # 对于缺失值会直接返回default_value
        normalizer_fn=lambda x: x / 6)
    # 注意下面的例子中，输入的数据和上面定义的解析规则并不一致。但是代码执行不会出错
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值必须为Tensor
        "feature": tf.constant(value=[
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]]
        ])
    })
    return column.get_dense_tensor(transformation_cache=feature_cache, state_manager=None)
```

# SequenceDenseColumn系列

这个是DenseColumn的序列形式。序列数据是允许有缺失值的。所以，SequenceDenseColumn的输入数据必须是SparseTensor。而DenseColumn的输入数据必须是Tensor。

## SequenceDenseColumn自定义方法

### get_sequence_dense_tensor(transformation_cache, state_manager)

这个方法返回一个TensorSequenceLengthPair(dense_tensor, sequence_length)。dense_tensor是序列数据的Tensor格式的输出，对于缺失的值会进行default_value填充。sequence_length是记录每个batch序列长度的Tensor，这里的序列长度是不包括最后连续填充的长度。

## sequence_numeric_column

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def sequence_numeric_column():
    # 用法同numeric_column基本一致
    # 区别在于两者输入数据的格式要求不同：这个只允许输入SparseTensor，上面的只允许输入Tensor
    # 由于这个方法输入是SparseTensor，所以shape参数是有意义的
    column = tf.feature_column.sequence_numeric_column(
        key="feature",
        # shape指定序列中每个元素的形状
        # 最终返回结构的形状为[batch_size, element_count/sum(shape[:]) ,shape]
        # 该值的设置只会影响dense_tensor。sequence_length只和实际输入数据有关
        shape=(3,),
        default_value=60,
        dtype=tf.float32,
        normalizer_fn=lambda x: x / 6)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # feature对应的值必须为SparseTensor
        "feature": tf.SparseTensor(
            # indices要按顺序写
            indices=[
                [0, 0, 1],
                [0, 1, 0],
                [0, 5, 0],
                [0, 5, 1],
                [1, 2, 1],
                [1, 3, 0],
                [1, 3, 1]
            ], 
            values=[4, 1, 7, 9, 3, 4., 4], 
            dense_shape=[2, 6, 2])
    })
    return column.get_sequence_dense_tensor(transformation_cache=feature_cache, state_manager=None)
```

# CategoricalColumn X DenseColumn 系列

该系列只有一个方法`bucketized_column`。这个方法需要传入一个`1-D`的`NumericColumn`（注意这里肯定是不支持`SequenceNumericColumn`的）。
其功能就是利用`boundaries`，将数值特征进行离散化。
例如，`boundaries=[3, 5, 7, 10]`，就会得到下表的映射规则：
| 规则 | 映射值 |
|:--------:|:-------:|
| x < 3| 1 0 0 0 0 |
| 3 <= x < 5| 0 1 0 0 0 |
| 5 <= x < 7 | 0 0 1 0 0 |
| 7 <= x < 10 | 0 0 0 1 0 |
| 10 <= x | 0 0 0 0 1 |

## bucketized_column的两个输出方法

### get_dense_tensor(transformation_cache, state_manager)

这个方法是从`DenseColumn`继承来的。返回的值是每个数值通过映射之后的one-hot的表示形式，就是上面映射规则中的映射值。

### get_sparse_tensors(transformation_cache, state_manager)

这个方法是从`CategoricalColumn`继承来的。返回的值是`get_dense_tensor`的输出在每个`batch`做`flatten`之后`one-hot`编码的`SparseTensor`格式。

## bucketized_column

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def bucketized_column():
    numeric_column = tf.feature_column.numeric_column(
        key="feature",
        shape=6,
        default_value=0,
        dtype=tf.float32)
    column = tf.feature_column.bucketized_column(
        # 1-D的numeric column
        source_column=numeric_column,
        # 要求列表为升序
        boundaries=[3, 5, 7, 10])
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        # 由于输入是从NumericColumn进入的，所以要求必须为Tensor
        # 注意这里的输入并不是1-D的，对于这种情况：
        # get_dense_tensor返回值中包含输入的shape信息
        # get_sparse_tensors会对输入数据进行flatten操作
        "feature": tf.constant(value=[
            [[1, 2], [3, 4], [5, 6]],
            [[7, 7], [9, 10], [11, 12]]
        ])
    })
    dense_tensor = column.get_dense_tensor(transformation_cache=feature_cache, state_manager=None)
    sparse_tensors = column.get_sparse_tensors(transformation_cache=feature_cache, state_manager=None)
    return dense_tensor, sparse_tensors
```

# DenseColumn X SequenceDenseColumn 系列

该系列下的方法都是利用`CategoricalColumn`作为输入数据的入口。由于继承了`DenseColumn`和`SequenceDenseColumn`，所以对非序列和序列特征都是支持的。
但是对于继承而来的两个方法`get_dense_tensor`和`get_sequence_dense_tensor`在使用时，需要看传入的`CategoricalColumn`是哪一种。
如果使用`categorical_*`方法创建的`CategoricalColumn`，需要调用`get_dense_tensor`。
如果使用`sequence_categorical_*`方法创建的`CategoricalColumn`，需要调用`get_sequence_dense_tensor`。
注意这两个方法不会同时请作用。

## indicator_column的两个输出方法

### get_dense_tensor(transformation_cache, state_manager)

这个方法是从`DenseColumn`继承来的。返回值计算的具体逻辑如下：
1. 通过`CategoricalColumn`获得输入数据的映射值，得到一个`shape=[a,b,c,d]`的`SparseTensor`
2. 根据`CategoricalColumn`的`num_buckets=N`值，将上面的`SparseTensor`转成`shape=[a,b,c,d,N]`的`Tensor`
3. 将上面的`Tensor`以`shape[-2]`为聚合维度进行`reduce_sum`操作，得到一个`shape=[a,b,c,N]`的`Tensor`
4. 返回上面的`Tensor`

### get_sequence_dense_tensor(transformation_cache, state_manager)

这个方法是从`SequenceDenseColumn`继承来的。这个方法返回一个`TensorSequenceLengthPair(dense_tensor, sequence_length)`。
`dense_tensor`的逻辑和`get_dense_tensor`一样。不过需要注意的是`SequenceColumn`对应的数据永远是`3-D`的，分别为`[batch, sequence, element]`。
`sequence_length`记录每个`batch`序列长度的`Tensor`，这里的序列长度是不包括最后连续填充的长度。

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def dense_indicator_column():
    categorical_column = tf.feature_column.categorical_column_with_identity(
        key="feature",
        num_buckets=5,
        default_value=0)
    # multi-hot编码，出现多次的相同值，会累加
    column = tf.feature_column.indicator_column(categorical_column=categorical_column)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [[[1, 2]], [[3, 4]], [[5, 5]]],
            [[[9, 8]], [[7, 6]], [[5, 4]]]
        ])
    })
    return column.get_dense_tensor(transformation_cache=feature_cache, state_manager=None)

def dense_sequence_indicator_column():
    sequence_categorical_column = tf.feature_column.sequence_categorical_column_with_identity(
        key="feature",
        num_buckets=5,
        default_value=0)
    # 当categorical_column为sequence类型时，返回值会多一个sequence_length，其余都一样
    column = tf.feature_column.indicator_column(categorical_column=sequence_categorical_column)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [[[1, 2]], [[3, 4]], [[5, 5]]],
            [[[9, 8]], [[7, 6]], [[5, 4]]]
        ])
    })
    return column.get_sequence_dense_tensor(transformation_cache=feature_cache, state_manager=None)
```

## embedding_column的两个输出方法

`EmbeddingColumn`(包括后面的`SharedEmbeddingColumn`)与前面的`Column`都不太一样。它是一种具有状态的`Column`，其状态数据通过`StateManager`进行管理。
这里所说的状态，其实就是模型需要进行训练的参数。`FeatureColumn`基类中有个`create_state`方法，就是用来进行参数初始化的。所以，对于具有状态的`Column`，需要调用`create_state`来初始化参数，才能进行后续的操作。

### get_dense_tensor(transformation_cache, state_manager)

这个方法是从`DenseColumn`继承来的。实现逻辑可以参考上面的`indicator_column`。两者的区别在于，`embedding_column`相较于`indicator_column`会在进行`one-hot`编码之后，不会直接返回这个编码，而是会将`one-hot`编码和`StateManager`管理的嵌入矩阵进行矩阵相乘，从而得到一个稠密的维度为`dimension`的向量。最终返回的值是这个稠密的向量。

### get_sequence_dense_tensor(transformation_cache, state_manager)

这个方法是从`SequenceDenseColumn`继承来的。这个和`indicator_column`是一样的逻辑。

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def embedding_column():
    categorical_column = tf.feature_column.categorical_column_with_identity(
        key="feature",
        num_buckets=5,
        default_value=0)
    weighted_categorical_column = tf.feature_column.weighted_categorical_column(
        categorical_column=categorical_column,
        weight_feature_key="feature_weights",
        dtype=tf.float32)
    column = tf.feature_column.embedding_column(
        categorical_column=weighted_categorical_column,
        # 嵌入权重的维度
        dimension=10,
        # 出现次数多次的值对应权重的聚合方式
        # 包括sum（加权和）、mean（加权和/总权重）、sqrtn（加权和/权重平方和的平方根）
        combiner="sqrtn",
        # 权重初始化方法
        initializer=tf.initializers.ones,
        # 指定从checkpoint文件里加载权重值的文件地址
        ckpt_to_load_from=None,
        # checkpoint中特征对应的名称
        tensor_name_in_ckpt=None,
        # 这个值参见：tf.clip_by_norm(tensor, max_norm)
        max_norm=None,
        # 权重是否需要训练
        trainable=True)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [[[1, 2]], [[3, 4]], [[5, 5]]],
            [[[9, 8]], [[7, 6]], [[5, 4]]]
        ]),
        "feature_weights": tf.constant(value=[
            [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]],
            [[9.9, 8.8, 7.7, 6.6, 5.5, 4.4]]
        ])
    })
    state_manager = feature_column_v2._StateManagerImplV2(layer=tf.keras.layers.Layer(), trainable=True)
    column.create_state(state_manager=state_manager)
    return column.get_dense_tensor(transformation_cache=feature_cache, state_manager=state_manager)

def sequence_embedding_column():
    sequence_categorical_column = tf.feature_column.sequence_categorical_column_with_identity(
        key="feature",
        num_buckets=5,
        default_value=0)
    # 用法和上面的完全一致，区别在于sequence的特征列在返回的时候，多一个sequence_length
    # 目前没有对应的带权SequenceCategoricalColumn实现，所以，只能支持不带权的CategoricalColumn
    column = tf.feature_column.embedding_column(
        categorical_column=sequence_categorical_column,
        dimension=10,
        combiner="mean",
        initializer=tf.initializers.glorot_normal,
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True)
    feature_cache = feature_column_lib.FeatureTransformationCache(features={
        "feature": tf.constant(value=[
            [[[1, 2]], [[3, 4]], [[5, 5]]],
            [[[9, 8]], [[7, 6]], [[5, 4]]]
        ])
    })
    state_manager = feature_column_v2._StateManagerImplV2(layer=tf.keras.layers.Layer(), trainable=True)
    column.create_state(state_manager=state_manager)
    return column.get_sequence_dense_tensor(transformation_cache=feature_cache, state_manager=state_manager)
```

## shared_embedding_column

实现逻辑与`EmbeddingColumn`相同。使用的时候，通过指定一个`CategoricalColumn`列表，得到一个`EmbeddingColumn`的列表。这个列表中的`EmbeddingColumn`共享同一份嵌入向量。

```python
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib

def shared_embedding_column():
    column1 = ...
    column2 = ...
    # 用法和embedding_columns基本一致
    columns = tf.feature_column.shared_embeddings(
        # 要求列表里的特征列，除了特征名称，其他属性必须完全一样
        categorical_columns=[column1, column2],
        # 指定共享嵌入向量的集合名称，这个不指定会提供默认值
        shared_embedding_collection_name=None,
        dimension=10,
        combiner="mean",
        initializer=tf.initializers.glorot_normal,
        ckpt_to_load_from=None,
        tensor_name_in_ckpt=None,
        max_norm=None,
        trainable=True)
    shared_column1, shared_column2 = columns
    return shared_column1, shared_column2
```
