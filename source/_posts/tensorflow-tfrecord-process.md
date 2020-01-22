---
title: 【Tensorflow2】TFRecord数据读写
mathjax: true
toc: true
comments: true
date: 2019-12-25 14:19:20
categories: tensorflow
tags: tensorflow tfrecord sequenceExample spark feature
---

TFRecord格式是官方推荐使用的模型输入数据的存储格式。模型在训练和预测的过程中，TFRecord格式用于组织模型的输入数据部分。

我们使用TFRecord来存储训练数据。同时，也使用TFRecord的数据作为模型计算的输入。

TFRecord的存储方式，能够以较小的空间来实现数据携带。对于基于Tensorflow Serving这种网络传输输入数据的打分形式，较小的传输数据，对打分性能的提升是很有帮助的。

所以，使用TFRecord就对了！

<!--more-->

## 样例数据

```
valueA1,2.3,valueC3:valueC2:valueC8,3:4,valueE3:valueE8:valueE3:valueE9,4.5:1.2:2.1,valueG5:valueG9#valueG3:valueG1#valueG5:valueG3,4:5:2#1:2:3,3:20:5:3,1:0:2:2,4:2:8:9,valueI6:valueI9:valueI3:valueI6
valueA2,4.1,valueC1:valueC2:valueC3,,valueE6:valueE1,9.4:6.6:8.3:7.2:9.1,valueG2:valueG1#valueG6:valueG6,1:1:3#4:2:9#8:4:2,5:10:2:2:6,4:2:7:6:3,3:1:8:4:2,valueI3:valueI5:valueI2:valueI7:valueI5
valueA1,3.7,valueC5:valueC5:valueC5,2:5,valueE3:valueE3:valueE3,5.3,,7:3:2#6:4:6#3:1:1#8:9:8,6:10,2:7,1:4,valueI5:valueI5
```

逗号（,）分割了每个特征列。井号（#）分割了每个特征内的多个取值。冒号（:）分割了每个取值的每个元素。
这里定义特征列的名称为：
["featureA", "featureB", "featureC", "featureD", "featureE", "featureF", "featureG", 
"featureH", "featureI_Index0", "featureI_Index1", "featureI_Index2", "featureI_value"]

## Schema定义

```python
import tensorflow as tf

column_schema = {
    # featureA: 一维字符串特征
    "featureA": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value="null"),
    # featureB: 一维数值特征
    "featureB": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0.0),
    # featureC: 三维字符串特征
    "featureC": tf.io.FixedLenFeature(shape=(3,), dtype=tf.string, default_value=["null", "null", "null"]),
    # featureD: 二维数值特征
    "featureD": tf.io.FixedLenFeature(shape=(2,), dtype=tf.int64, default_value=[0, 0]),
    # featureE: 不固定维度字符串特征
    "featureE": tf.io.VarLenFeature(dtype=tf.string),
    # featureF: 不固定维度数值特征
    "featureF": tf.io.VarLenFeature(dtype=tf.float32),
    # featureG: 二维字符串序列特征
    "featureG": tf.io.FixedLenSequenceFeature(shape=(2,), dtype=tf.string, allow_missing=True, default_value=None),
    # featureH: 三维数值序列特征
    "featureH": tf.io.FixedLenSequenceFeature(shape=(3,), dtype=tf.int64, allow_missing=True, default_value=None),
    # featureI: 21 * 4 * 10 维字符串稀疏特征
    "featureI": tf.io.SparseFeature(index_key=["featureI_Index0", "featureI_Index1", "featureI_Index2"],
                                    value_key="featureI_value", dtype=tf.string, size=[21, 4, 10], already_sorted=False)
}
```

> tf.io.FixedLenFeature

用于解析shape和类型确定的特征列。在特征列缺失的情况下，default_value不设置会引发错误。default_value的值需要和设置的shape保持一致。

> tf.io.VarLenFeature

用于解析类型确定，但是shape不确定的特征列。

> tf.io.FixedLenSequenceFeature

用于解析shape和类型确定的序列特征列。这里的序列特征存储的数据像这样：[(1,2), (3,4), (5,6)]。列表的长度不固定，但是每个元素的shape固定。

allow_missing=True允许特征值不存在。通过default_value来指定填充的默认值。

tf.io.parse_sequence_example方法貌似要求default_value必须设置为None

> tf.io.SparseFeature

用于解析通过稀疏矩阵的格式来存储的特征，包括index_key和value_key两部分。凡是用到SparseFeature的地方，都建议使用VarLenFeature来替代。

## TFRecord数据读写（java/scala版）

### 使用tensorflow-utils组装TFRecord数据

```java
import org.tensorflow.example.Example;
import org.tensorflow.example.SequenceExample;
import top.ritchie.tensorflow.utils.feature.FeatureUtils;
import top.ritchie.tensorflow.utils.feature.TFRecordGen;

/*
    git@github.com:ritchietop/tensorflow-utils.git
 */

public class ExampleGen {

    public static void main(String[] args) {
        Example example = new TFRecordGen(5)
                .put("featureA", FeatureUtils.byteStringFeature("valueA1"))
                .put("featureB", FeatureUtils.floatListFeature(2.3))
                .put("featureC", FeatureUtils.byteStringFeature("valueC3", "valueC2", "valueC8"))
                .put("featureD", FeatureUtils.int64ListFeature(3, 4))
                .put("featureE", FeatureUtils.byteStringFeature("valueE3", "valueE8", "valueE3", "valueE9"))
                .put("featureF", FeatureUtils.floatListFeature(4.5, 1.2, 2.1))
                .genExample();
        SequenceExample sequenceExample = new TFRecordGen(10, 2)
                .put("featureA", FeatureUtils.byteStringFeature("valueA2"))
                .put("featureB", FeatureUtils.floatListFeature(4.1))
                .put("featureC", FeatureUtils.byteStringFeature("valueC1", "valueC2", "valueC3"))
                .put("featureD", FeatureUtils.int64ListFeature())
                .put("featureE", FeatureUtils.byteStringFeature("valueE6", "valueE1"))
                .put("featureF", FeatureUtils.floatListFeature(9.4, 6.6, 8.3, 7.2, 9.1))
                .put("featureG", FeatureUtils.toFeatureList(
                    FeatureUtils.byteStringFeature("valueG2", "valueG1"),
                    FeatureUtils.byteStringFeature("valueG6", "valueG6")
                ))
                .put("featureH", FeatureUtils.toFeatureList(
                    FeatureUtils.int64ListFeature(1, 1, 3),
                    FeatureUtils.int64ListFeature(4, 2, 9),
                    FeatureUtils.int64ListFeature(8, 4, 2)
                ))
                .put("featureI_Index0", FeatureUtils.int64ListFeature(5, 10, 2, 2, 6))
                .put("featureI_Index1", FeatureUtils.int64ListFeature(4, 2, 7, 6, 3))
                .put("featureI_Index2", FeatureUtils.int64ListFeature(3, 1, 8, 4, 2))
                .put("featureI_value", FeatureUtils.byteStringFeature("valueI3", "valueI5", "valueI2", "valueI7", "valueI5"))
                .genSequenceExample();
    }

}
```

### 使用Spark快速读写TFRecord数据

```
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.{SparkConf, SparkContext}
import org.tensorflow.spark.shaded.org.tensorflow.hadoop.io.{TFRecordFileInputFormat, TFRecordFileOutputFormat}

/**
    <dependency>
        <groupId>org.tensorflow</groupId>
        <artifactId>spark-tensorflow-connector_2.11</artifactId>
        <version>1.15.0</version>
    </dependency>
 */

object ExampleGen {

    def main(args: Array[String]): Unit = {
        val sc = new SparkContext(new SparkConf().setAppName("ExampleGenTask").setMaster("local"))
        // TFRecord写入
        sc.makeRDD(Seq(example, sequenceExample)).repartition(1)
        .map(example => (new BytesWritable(example.toByteArray), NullWritable.get()))
        .saveAsNewAPIHadoopFile[TFRecordFileOutputFormat]("path/data")

        // TFRecord读取
        sc.newAPIHadoopFile("path/data", classOf[TFRecordFileInputFormat], classOf[BytesWritable], classOf[NullWritable])
        .map { case (bytesWritable, nullWritable) => SequenceExample.parseFrom(bytesWritable.getBytes) }
        .collect()
        .foreach(println)
    }

}
```

## TFRecord数据读写（python版）


### TFRecord写入样例

```python
import tensorflow as tf

writer = tf.io.TFRecordWriter("path/tfrecord")

example = tf.train.Example(features=tf.train.Features(feature={
    "featureA": tf.train.Feature(bytes_list=tf.train.BytesList(value=[u"valueA1".encode("utf-8")])),
    "featureB": tf.train.Feature(float_list=tf.train.FloatList(value=[2.3])),
    "featureC": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueC3", b"valueC2", b"valueC8"])),
    "featureD": tf.train.Feature(int64_list=tf.train.Int64List(value=[3, 4])),
    "featureE": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueE3", b"valueE8", b"valueE3", b"valueE9"])),
    "featureF": tf.train.Feature(float_list=tf.train.FloatList(value=[4.5, 1.2, 2.1]))
}))

sequence_example = tf.train.SequenceExample(
    context=tf.train.Features(feature={
        "featureA": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueA2"])),
        "featureB": tf.train.Feature(float_list=tf.train.FloatList(value=[4.1])),
        "featureC": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueC1", b"valueC2", b"valueC3"])),
        "featureE": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueE6", b"valueE1"])),
        "featureF": tf.train.Feature(float_list=tf.train.FloatList(value=[9.4, 6.6, 8.3, 7.2, 9.1])),
        "featureI_Index0": tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 10, 2, 2, 6])),
        "featureI_Index1": tf.train.Feature(int64_list=tf.train.Int64List(value=[4, 2, 7, 6, 3])),
        "featureI_Index2": tf.train.Feature(int64_list=tf.train.Int64List(value=[3, 1, 8, 4, 2])),
        "featureI_value": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueI3", b"valueI5", b"valueI2", b"valueI7", b"valueI5"]))
    }),
    feature_lists=tf.train.FeatureLists(feature_list={
        "featureG": tf.train.FeatureList(feature=[
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueG2", b"valueG1"])),
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"valueG6", u"valueG6".encode("utf-8")]))
        ]),
        "featureH": tf.train.FeatureList(feature=[
            tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 1, 3])),
            tf.train.Feature(int64_list=tf.train.Int64List(value=[4, 2, 9])),
            tf.train.Feature(int64_list=tf.train.Int64List(value=[8, 4, 2]))
        ])
    })
)

writer.write(example.SerializeToString())
writer.write(sequence_example.SerializeToString())

writer.close()
```

### TFRecord解析

使用tf.io.parse_single_sequence_example解析SequenceExample数据时，不支持tf.io.SparseFeature特征列，所以解析的时候把featureI剔除了。


```python
import tensorflow as tf

context_schema = {
    "featureA": column_schema["featureA"],
    "featureB": column_schema["featureB"],
    "featureC": column_schema["featureC"],
    "featureD": column_schema["featureD"],
    "featureE": column_schema["featureE"],
    "featureF": column_schema["featureF"],
    "featureI": column_schema["featureI"],
}

sequence_schema = {
    "featureG": column_schema["featureG"],
    "featureH": column_schema["featureH"],
}

# 这个是官方提供的高效加载外部数据的工具，具体使用可以参加tensorflow dataset
data = tf.data.TFRecordDataset(filenames="path/tfrecord")

for index, record in enumerate(data):
    if index == 0:
        example = tf.io.parse_single_example(record, features=context_schema)
        print("======================example======================")
        for key, value in example.items():
            print(key, "=>", value)
    else:
        # 剔除SparseFeature特征
        context_schema.pop("featureI")
        # 这里返回两个dict，分别对应contextFeatures和sequenceFeature
        (context, sequence) = tf.io.parse_single_sequence_example(
            record, context_features=context_schema, sequence_features=sequence_schema)
        print("==============sequenceExample: context==============")
        for key, value in context.items():
            print(key, "=>", value)
        print("==============sequenceExample: featureList==============")
        for key, value in sequence.items():
            print(key, "=>", value)
```

上面的程序解析的结果如下：

```
======================example======================
featureA => tf.Tensor([b'valueA1'], shape=(1,), dtype=string)
featureB => tf.Tensor([2.3], shape=(1,), dtype=float32)
featureC => tf.Tensor([b'valueC3' b'valueC2' b'valueC8'], shape=(3,), dtype=string)
featureD => tf.Tensor([3 4], shape=(2,), dtype=int64)
featureE => SparseTensor(indices=tf.Tensor(
[[0]
 [1]
 [2]
 [3]], shape=(4, 1), dtype=int64), values=tf.Tensor([b'valueE3' b'valueE8' b'valueE3' b'valueE9'], shape=(4,), dtype=string), dense_shape=tf.Tensor([4], shape=(1,), dtype=int64))
featureF => SparseTensor(indices=tf.Tensor(
[[0]
 [1]
 [2]], shape=(3, 1), dtype=int64), values=tf.Tensor([4.5 1.2 2.1], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3], shape=(1,), dtype=int64))
featureI => SparseTensor(indices=tf.Tensor([], shape=(0, 3), dtype=int64), values=tf.Tensor([], shape=(0,), dtype=string), dense_shape=tf.Tensor([21  4 10], shape=(3,), dtype=int64))
==============sequenceExample: context==============
featureA => tf.Tensor([b'valueA2'], shape=(1,), dtype=string)
featureB => tf.Tensor([4.1], shape=(1,), dtype=float32)
featureC => tf.Tensor([b'valueC1' b'valueC2' b'valueC3'], shape=(3,), dtype=string)
featureD => tf.Tensor([0 0], shape=(2,), dtype=int64)
featureE => SparseTensor(indices=tf.Tensor(
[[0]
 [1]], shape=(2, 1), dtype=int64), values=tf.Tensor([b'valueE6' b'valueE1'], shape=(2,), dtype=string), dense_shape=tf.Tensor([2], shape=(1,), dtype=int64))
featureF => SparseTensor(indices=tf.Tensor(
[[0]
 [1]
 [2]
 [3]
 [4]], shape=(5, 1), dtype=int64), values=tf.Tensor([9.4 6.6 8.3 7.2 9.1], shape=(5,), dtype=float32), dense_shape=tf.Tensor([5], shape=(1,), dtype=int64))
==============sequenceExample: featureList==============
featureG => tf.Tensor(
[[b'valueG2' b'valueG1']
 [b'valueG6' b'valueG6']], shape=(2, 2), dtype=string)
featureH => tf.Tensor(
[[1 1 3]
 [4 2 9]
 [8 4 2]], shape=(3, 3), dtype=int64)
```

#### Example解析API

- tf.io.parse_single_example(serialized, features)
- tf.io.parse_example(serialized, features)
- tf.io.parse_single_sequence_example(serialized, context_features=None, sequence_features=None)
- tf.io.parse_sequence_example(serialized, context_features=None, sequence_features=None)

single版解析单条数据，非single版需要加一个batch维度。其余方面两者用法完全一致。
