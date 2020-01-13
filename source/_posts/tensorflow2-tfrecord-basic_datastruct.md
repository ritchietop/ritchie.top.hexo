---
title: 【Tensorflow2】数据存储——TFRecord的数据结构
mathjax: true
toc: true
comments: true
date: 2019-12-24 17:33:06
categories: tensorflow
tags: tensorflow tfrecord sequenceExample
---



TFRecord是官方推荐使用的tensorflow模型数据存储格式。基于该格式的模型数据，可以实现较小空间大小的数据携带。
这里使用java的API，对TFRecord中基本数据结构：BytesList、FloatList、Int64List、Feature、Features、Example、FeatureList、FeatureLists、SequenceExample进行说明。

<!--more-->

## 关键数据结构的proto定义
```
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };

message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
  Features context = 1;
  FeatureLists feature_lists = 2;
};
```

## java代码中需要引入的数据类型
```java
import com.google.protobuf.ByteString;
import org.tensorflow.example.BytesList;
import org.tensorflow.example.FloatList;
import org.tensorflow.example.Int64List;
import org.tensorflow.example.Feature;
import org.tensorflow.example.Features;
import org.tensorflow.example.FeatureList;
import org.tensorflow.example.FeatureLists;
import org.tensorflow.example.Example;
import org.tensorflow.example.SequenceExample;
```

依赖包配置
```
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>proto</artifactId>
    <version>${tensorflow.version}</version>
</dependency>
```

## BytesList, FloatList, Int64List

这三种类型是TFRecord的基本数据结构，内部包装的是三种不同类型的列表，并且提供的操作，都是基于内部包装类型进行的。

BytesList是对`List<byteString>`类型的包装

```java
BytesList.Builder bytesListBuilder = BytesList.newBuilder();
BytesList bytesList = bytesListBuilder
        .addValue(ByteString.copyFromUtf8("A"))
        .addValue(ByteString.copyFromUtf8("B"))
        .setValue(0, ByteString.copyFromUtf8("C"))
        .addValue(ByteString.copyFromUtf8("D"))
        .build();
System.out.println(bytesList);
/*
===========output===========
value: "C"
value: "B"
value: "D"
============================
*/
```

FloatList是对`List<float32>`类型的包装

```java
FloatList.Builder floatListBuilder = FloatList.newBuilder();
FloatList floatList = floatListBuilder
        .addValue(1F).addValue(2F).addValue(3F)
        .mergeFrom(FloatList.newBuilder().addValue(4F).addValue(5F).build())
        .build();
System.out.println(floatList);
/*
===========output===========
value: 1.0
value: 2.0
value: 3.0
value: 4.0
value: 5.0
============================
*/
```

Int64List是对`List<int64>`类型的包装

```java
Int64List.Builder int64ListBuilder = Int64List.newBuilder();
Int64List int64List = int64ListBuilder
        .addAllValue(Arrays.asList(1L, 2L, 3L, 4L))
        .build();
System.out.println(int64List);
/*
===========output===========
value: 1
value: 2
value: 3
value: 4
============================
*/
```

## Feature, FeatureList

Feature是对BytesList，Int64List，FloatList三种类型中的一种进行了包装。通过Feature的包装，隐藏了不同特征列的类型差异。

FeatureList是对`List<Feature>`的包装。每个Feature包装的list（bytesList, Int64List, FloatList）长度可以不同。

```java
// feature中只能包含一种list，多次赋值会被覆盖
Feature.Builder featureBuilder = Feature.newBuilder();
Feature feature = featureBuilder
        .setBytesList(bytesList)
        .setInt64List(int64List)
        .setFloatList(floatList)
        .build();
System.out.println(feature);
/*
===========output===========
float_list {
  value: 1.0
  value: 2.0
  value: 3.0
  value: 4.0
  value: 5.0
}
============================
*/
```

```java
FeatureList.Builder featureListBuilder = FeatureList.newBuilder();
FeatureList featureList = featureListBuilder
        .addFeature(feature)
        .addFeature(Feature.newBuilder().setInt64List(int64List).build())
        .build();
System.out.println(featureList);
/*
===========output===========
feature {
  float_list {
    value: 1.0
    value: 2.0
    value: 3.0
    value: 4.0
    value: 5.0
  }
}
feature {
  int64_list {
    value: 1
    value: 2
    value: 3
    value: 4
  }
}
============================
*/
```

## Features, FeatureLists

Features是对`Map<String, Feature>`的包装。

FeatureLists是对`Map<String, FeatureList>`的包装。

不同的key对应的不同的特征名称。

```java
Feature.Builder bytesFeatureBuilder = Feature.newBuilder();
Feature bytesFeature = bytesFeatureBuilder.setBytesList(bytesList).build();
Feature.Builder floatFeatureBuilder = Feature.newBuilder();
Feature floatFeature = floatFeatureBuilder.setFloatList(floatList).build();
Features.Builder featuresBuilder = Features.newBuilder();
Features features = featuresBuilder
        .putFeature("bytesFeatureKey", bytesFeature)
        .putFeature("floatFeatureKey", floatFeature)
        .putFeature("emptyFeatureKey", Feature.newBuilder().build())
        .build();
System.out.println(features);
/*
===========output===========
feature {
  key: "bytesFeatureKey"
  value {
    bytes_list {
      value: "C"
      value: "B"
      value: "D"
    }
  }
}
feature {
  key: "floatFeatureKey"
  value {
    float_list {
      value: 1.0
      value: 2.0
      value: 3.0
      value: 4.0
      value: 5.0
    }
  }
}
feature {
  key: "emptyFeatureKey"
  value {
  }
}
============================
*/
```

```java
FeatureLists.Builder featureListsBuilder = FeatureLists.newBuilder();
FeatureLists featureLists = featureListsBuilder
        .putAllFeatureList(new HashMap<String, FeatureList>() {{
            put("featureListKey", featureList);
            put("emptyFeatureListKey", FeatureList.newBuilder().build());
        }})
        .build();
System.out.println(featureLists);
/*
===========output===========
feature_list {
  key: "featureListKey"
  value {
    feature {
      float_list {
        value: 1.0
        value: 2.0
        value: 3.0
        value: 4.0
        value: 5.0
      }
    }
    feature {
      int64_list {
        value: 1
        value: 2
        value: 3
        value: 4
      }
    }
  }
}
feature_list {
  key: "emptyFeatureListKey"
  value {
  }
}
============================
*/
```

## Example, SequenceExample

Example和SequenceExample是TFRecord最终序列化的两种格式。

Example是对Features的包装。

SequenceExample是对Features和FeatureLists的包装。

```java
Example.Builder exampleBuilder = Example.newBuilder();
Example example = exampleBuilder
        .setFeatures(features)
        .build();
System.out.println(example);
/*
===========output===========
features {
  feature {
    key: "bytesFeatureKey"
    value {
      bytes_list {
        value: "C"
        value: "B"
        value: "D"
      }
    }
  }
  feature {
    key: "floatFeatureKey"
    value {
      float_list {
        value: 1.0
        value: 2.0
        value: 3.0
        value: 4.0
        value: 5.0
      }
    }
  }
  feature {
    key: "emptyFeatureKey"
    value {
    }
  }
}
============================
*/
```

```java
SequenceExample.Builder sequenceExampleBuilder = SequenceExample.newBuilder();
SequenceExample sequenceExample = sequenceExampleBuilder
        .setContext(features)
        .setFeatureLists(featureLists)
        .build();
System.out.println(sequenceExample);
/*
===========output===========
context {
  feature {
    key: "bytesFeatureKey"
    value {
      bytes_list {
        value: "C"
        value: "B"
        value: "D"
      }
    }
  }
  feature {
    key: "floatFeatureKey"
    value {
      float_list {
        value: 1.0
        value: 2.0
        value: 3.0
        value: 4.0
        value: 5.0
      }
    }
  }
  feature {
    key: "emptyFeatureKey"
    value {
    }
  }
}
feature_lists {
  feature_list {
    key: "featureListKey"
    value {
      feature {
        float_list {
          value: 1.0
          value: 2.0
          value: 3.0
          value: 4.0
          value: 5.0
        }
      }
      feature {
        int64_list {
          value: 1
          value: 2
          value: 3
          value: 4
        }
      }
    }
  }
  feature_list {
    key: "emptyFeatureListKey"
    value {
    }
  }
}
============================
*/
```
