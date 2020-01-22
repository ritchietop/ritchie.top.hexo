---
title: 【Tensorflow2】Dataset简明教程
mathjax: true
toc: false
comments: true
date: 2019-12-28 14:18:10
categories: tensorflow
tags: tensorflow dataset pipeline
---

Dataset是Tensorflow提供的一套高效的数据加载工具。我们可以利用Dataset以简单可复用的方法构建复杂的输入Pipeline。

自Tensorflow2.0之后，Dataset的API变得更加的简单易用。用过的Tensorflow1.x的朋友可能会知道，Dataset之前需要依赖Iterator来实现数据迭代。为了提供Tensorflow的易用性，2.0版本开始，Dataset可以直接使用for语句来实现数据遍历，使得Dataset更加的pythonic。

简单、方便、强大，你不选择使用Dataset，我不信。

<!--more-->

## Dataset创建

### 创建单元素dataset
```python
dataset_single = tf.data.Dataset.from_tensors([[1], [2], [3], [4], [5]])
for element in dataset_single:
    print(element)
print(dataset_single.element_spec)  # TensorSpec(shape=(5, 1), dtype=tf.int32, name=None)
```

### 创建多元素dataset
```python
dataset_multi = tf.data.Dataset.from_tensor_slices([[1], [2], [3], [4], [5]])
for element in dataset_multi:
    print(element)
print(dataset_multi.element_spec)  # TensorSpec(shape=(1,), dtype=tf.int32, name=None)
```

### 从生成器创建dataset
```python
def generator(init_value, max_step):
    i = init_value
    while i < max_step:
        yield [[i + 1], [i + 2], [i + 3]], [i]
        i *= 2
dataset_gen = tf.data.Dataset.from_generator(
    generator=generator, args=[2, 10], output_types=(tf.float32, tf.int64), output_shapes=((3, 1), (1,)))
for element in dataset_gen:
    print(element)
print(dataset_gen.element_spec)  # TensorSpec(shape=(3, 1), dtype=tf.float32, name=None)
```

### 仿range创建dataset
```python
dataset_range = tf.data.Dataset.range(10, 100, 5)
for element in dataset_range:
    print(element)
print(dataset_range.element_spec)  # TensorSpec(shape=(), dtype=tf.int64, name=None)
```

### 路径正则创建dataset

根据指定的文件路径样式，列出所有匹配的路径。通过shuffle确定是否打散输出结果

```python
dataset_files = tf.data.Dataset.list_files(file_pattern="path/*.py", shuffle=None, seed=None)
for element in dataset_files:
    print(element)
print(dataset_files.element_spec)
```

### 仿zip创建dataset
```python
dataset_zip = tf.data.Dataset.zip(datasets=[tf.data.Dataset.range(3), tf.data.Dataset.range(4, 7)])
for element in dataset_zip:
    print(element)
print(dataset_zip.element_spec)
```

### 从文件创建dataset

> 行文本格式
```python
text_line_path = "path/text"
dataset_text = tf.data.TextLineDataset(
    filenames=[text_line_path], compression_type="", buffer_size=None, num_parallel_reads=None)
for line in dataset_text:
    print(line)
print(dataset_text.element_spec)  # TensorSpec(shape=(), dtype=tf.string, name=None)
```

> TFRecord格式
```python
tfrecord_path = "path/tfrecord"
dataset_tfrecord = tf.data.TFRecordDataset(
    filenames=[tfrecord_path], compression_type="", buffer_size=None, num_parallel_reads=None)
for record in dataset_tfrecord:
    example = tf.io.parse_single_example(record, features={"A": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64)})
    print(example)
```

> 固定长度bytes格式
```python
binary_path = "path/binary"
# 这里要求文件的总bytes数在减去header_bytes和footer_bytes忽略的长度之后，需要是record_bytes的整数倍
dataset_binary = tf.data.FixedLengthRecordDataset(
    filenames=[binary_path], record_bytes=10, header_bytes=6, footer_bytes=4, buffer_size=None,
    compression_type="", num_parallel_reads=None)
for bytes_record in dataset_binary:
    print(bytes_record)
```

## Dataset操作

在介绍下面的函数之前，需要明确的一点是，所有的操作所处理的元素，均表示的是该操作上一步每次迭代所返回的数据单元。

为了更好的理解每个函数的含义，明确其所处理的元素具体指的是什么，是一件很重要的事情。

- dataset.enumerate(start: int)

    为记录添加index（包含start），记录变成了tuple类型(index_tensor, tensor)

- dataset.take(count: int)

    截取count个记录

- dataset.apply(transformation_func=None)
    
    对整个数据集应用transformation_func
    
- dataset.batch(batch_size: int, drop_remainder: bool)

    为数据增加batch维度，如果总条数不能整除batch_size，通过drop_remainder确定是否舍弃末尾记录
    
- dataset.cache(filename: str)
    
    将数据缓存到指定filename中，不指定filename，表示缓存到内存中

- dataset.concatenate(dataset: Dataset)

    数据集串联合并，将参数中的dataset拼接到原dataset的后面
    
- dataset.filter(predicate: => bool)

    过滤数据，predicate需要返回bool值，True表示不过滤，False表示过滤
    
- dataset.flat_map(map_func=None)

    同map，不过会将结果展开

- dataset.interleave(map_func: => Dataset, cycle_length: int, block_length: int, num_parallel_calls: int)

    - map_func: 将dataset中的每个元素转换成一个Dataset
    - cycle_lenght: 设置同时处理dataset中元素的个数，即每次同时读取多少个dataset中的元素执行map_func方法。默认设置为当前可用的cpu核数
    - block_length: 每个元素产生的Dataset每次输出的元素个数
    - num_parallel_calls: 设置并发度，默认不并发执行

    这个方法的执行逻辑，我用队列的思路模拟一下。
    执行步骤：
    1. 从dataset中读取cycle_length个元素
    2. 调度map_func方法，产生cycle_length个Dataset
    3. 把cycle_length个Dataset放到一个队列里
    4. 先读取队首的Dataset，遍历输出block_length个元素，然后，将这个Dataset放到队列的末尾
    5. 循环执行4的步骤，直到每个Dataset中的元素可能不足block_length个，那就有多少算多少的输出
    6. 到这里从第1步中读取的cycle_length个元素就遍历完成了
    7. 回到第1步重新开始下一轮

    ```python
        dataset = tf.data.Dataset.range(5)
        dataset = dataset.interleave(
            map_func=lambda x: tf.data.Dataset.from_tensors(x).repeat(6),
            cycle_length=3,
            block_length=4)
        for record in dataset:
            print(record)
    ```

    Output：[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4]

    可以仿照上面的步骤，推演一下这个代码的执行过程。

- dataset.map(map_func=None, num_parallel_calls: int)

    功能同python的map。为每条记录应用map_func方法，通过num_parallel_calls设置并发度，None表示串行执行。

- dataset.options()

    返回数据集和输入的option

- dataset.padded_batch(batch_size: int, padded_shapes, padding_values=None, drop_remainder: bool)

    - batch_size: 设置批次大小
    - padded_shapes: 根据dataset中的记录特征，为每个维度设置扩充的最大维度大小，None表示使用当前批次内所以记录的最大维度大小
    - padding_values: 设置填充的数值，这个值需要为标量，需要注意的是要保持数值类型与dataset数值类型的一致性
    - drop_remainder: 剩余记录不足一个批次时，是否输出

    使用batch方法的时候，需要dataset中所有记录都具有相同的shape。如果我们的dataset中存储的记录的shape并不是相同的，就可以使用padded_batch实现batch的效果

    她的实现方法是对需要batch在一起的shape不同的记录，通过padded_shapes在各维度指定的扩充维度进行扩充，填充的值通过padding_values指定

    padded_batch能够实现不同shape记录的batch。但是，dataset每次输出数据的shape并不一定相同

    ```python
        dataset = tf.data.Dataset.range(5).map(lambda tensor: tf.fill([tensor, tensor], tf.cast(tensor, dtype=tf.int32)))
        dataset = dataset.padded_batch(3, padded_shapes=[None, 4], padding_values=tf.constant(value=-1, dtype=tf.int32))
        for record in dataset:
            print(record)
    ```

    执行结果：

    ```text
    tf.Tensor(
    [[[-1 -1 -1 -1]
    [-1 -1 -1 -1]]

    [[ 1 -1 -1 -1]
    [-1 -1 -1 -1]]

    [[ 2  2 -1 -1]
    [ 2  2 -1 -1]]], shape=(3, 2, 4), dtype=int32)

    tf.Tensor(
    [[[ 3  3  3 -1]
    [ 3  3  3 -1]
    [ 3  3  3 -1]
    [-1 -1 -1 -1]]

    [[ 4  4  4  4]
    [ 4  4  4  4]
    [ 4  4  4  4]
    [ 4  4  4  4]]], shape=(2, 4, 4), dtype=int32)
    ```

- dataset.prefetch(buffer_size: int)

    开启数据预读取机制。buffer_size设置缓冲区的大小，设置为None或者-1，表示自动配置缓冲区大小

- dataset.reduce(initial_state=None, reduce_func=None)
    
    功能同python的reduce。将每条记录通过reduce_func依次执行，最终返回一条记录。

- dataset.repeat(count: int)

    数据集重复迭代次数。count设置为None或者-1，表示迭代次数为无穷。count设置为1的效果和不调用repeat一致
    
- dataset.shard(num_shards: int, index: int)

    多work执行时，用该方法可以将数据集切割为num_shards份，index表示work_index

    通常num_shards > work_count，通过work_index可以为每个work分配若干个碎片的dataset
    
- dataset.shuffle(buffer_size: int, seed=None, reshuffle_each_iteration: bool)

    - buffer_size: 随机缓冲区的空间大小
    - seed: 随机种子
    - reshuffle_each_iteration: 每个迭代是否重新混洗，默认为True

    随机的策略：
    1. 从dataset中读取buffer_size个元素放到缓冲区中
    2. 从缓冲区中随机选择一个元素输出
    3. 再从dataset补充一个元素放到缓冲区中，维持buffer_size个元素
    4. 回到第2步

    为了达到更好的shuffle效果，shuffle的缓冲区大小应该不小于整个dataset的大小，即在整个dataset上进行随机（当然，也需要考虑dataset的大小）。

    ```python
        dataset = tf.data.Dataset.range(10)
        dataset = dataset.shuffle(buffer_size=2, seed=5, reshuffle_each_iteration=False)
        for record in dataset:
            print(record)
    ```

    Output：[1, 2, 0, 4, 3, 5, 7, 8, 6, 9]

- dataset.skip(count: int)

    跳过count条记录，count设置为-1，表示跳过之后的所有记录

- dataset.unbatch()

    batch和padded_batch的逆操作，将一个批次为N的记录，分成N条记录。

- dataset.window(size=1, shift=None, stride=1, drop_remainder=False)

    - size: 每个dataset的最大元素个数
    - shift: 每个dataset首元素的步长
    - stride: dataset内部元素的步长
    - drop_remainder: dataset的长度不足size时，是否舍弃

    ```python
        dataset = tf.data.Dataset.from_tensor_slices(["a", "b", "c", "d", "e", "f", "g", "h"])
        dataset = dataset.window(size=3, shift=2, stride=3, drop_remainder=False)
        for record in dataset:
            for element in record:
                print(element)
            print()
    ```

    执行结果：

    ```
    tf.Tensor(b'a', shape=(), dtype=string)
    tf.Tensor(b'd', shape=(), dtype=string)
    tf.Tensor(b'g', shape=(), dtype=string)

    tf.Tensor(b'c', shape=(), dtype=string)
    tf.Tensor(b'f', shape=(), dtype=string)

    tf.Tensor(b'e', shape=(), dtype=string)
    tf.Tensor(b'h', shape=(), dtype=string)

    tf.Tensor(b'g', shape=(), dtype=string)
    ```

    size控制了每个dataset的最大元素个数。上面代码返回了4个dataset，每个dataset的最大长度都不超过size=3

    shift设置每个dataset首元素的间隔。上面dataset的首元素分别为["a", "c", "e", "g"]，相邻两个dataset之间的步长都是shift=2

    stride设置了每个dataset内部元素之间的步长。第一个dataset=["a", "d", "g"]，相邻两个元素之间的步长是stride=3。后面的几个dataset都是这样的

    drop_remainder设置为True，则元素个数少于size的dataset，都会被舍弃

- dataset.with_options(options=None)

    为dataset设置options

## 高效流水线

- 使用Prefetch数据预加载机制
- 使用Interleave并发
- 使用map操作时，设置num_parallel_calls参数
- 在数据第一次epoch时，使用cache操作来缓存数据
- 使用map时，设置向量化的函数
- 使用interleave、prefetch、shuffle前，先使用能够减少数据空间的操作

## 好用的API

- tf.data.experimental.make_batched_features_dataset

    这个函数可以生成指定批次的数据集。这个函数只是对Example格式存储的TFRecord数据进行解析

    - file_pattern: 设置文件路径
    - batch_size: 批次大小
    - features: TFRecord的schema
    - reader=core_readers.TFRecordDataset: reader
    - label_key=None: 指定label的名称
    - reader_args=None: reader args
    - num_epochs=None: 数据集迭代轮数
    - shuffle=True: 是否shuffle
    - shuffle_buffer_size=10000: shuffle buffer size
    - shuffle_seed=None: shuffle size
    - prefetch_buffer_size=dataset_ops.AUTOTUNE: prefetch buffer size
    - reader_num_threads=1: 这个赋值给了parallel_interleave函数的cycle_length
    - parser_num_threads=2: 这个赋值给了parse_example_dataset函数的num_parallel_calls
    - sloppy_ordering=False: 这个赋值给了parallel_interleave函数的sloppy
    - drop_final_batch=False: 这个赋值给了batch函数的drop_remainder

    parallel_interleave这个函数是tensorflow1.x提供的，现在已经不建议使用。可以理解和interleave的效果一致，sloppy_ordering设置为False就可以。

- tf.data.experimental.make_csv_dataset

    这个函数和上面类似，不过是处理CSV格式的数据。

    - file_pattern: 设置文件路径
    - batch_size: 批次大小
    - column_names=None: 列名
    - column_defaults=None: 列默认值
    - label_name=None: 指定label的名称
    - select_columns=None: 
    - field_delim=",": 列分隔符
    - use_quote_delim=True: 
    - na_value="": 
    - header=True: 
    - num_epochs=None: 
    - shuffle=True: 
    - shuffle_buffer_size=10000: 
    - shuffle_seed=None: 
    - prefetch_buffer_size=dataset_ops.AUTOTUNE: 
    - num_parallel_reads=1: 
    - sloppy=False: 
    - num_rows_for_inference=100: 
    - compression_type=None: 
    - ignore_errors=False: 
