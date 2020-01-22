---
title: 【Tensorflow2】Estimator定制化流程
mathjax: true
toc: false
comments: true
date: 2020-01-22 16:33:21
categories: tensorflow
tags: estimator head flag keras exporter dataset
---

借助Tensorflow实现自定义模型的方法有很多。通过参考官方给出的LR、DNN、WideAndDeep模型的实现，我总结了一套较为完整的使用Estimator自定义模型的流程。
这里并不涉及具体的实现细节，只有整体的架构设计。细节部分后续会再分享。

<!--more-->

# 1. 自定义Model的基本架构（以YouTube DNN模型的双塔结构为例）

```python
class UserDefineModel(tf.keras.Model):
    def __init__(self,
                 user_column_features: list,
                 item_column_features: list,
                 user_hidden_units: list,
                 item_hidden_units: list,
                 user_activation: Callable[[tf.Tensor], tf.Tensor],
                 item_activation: Callable[[tf.Tensor], tf.Tensor],
                 user_dropout: float,
                 item_dropout: float,
                 user_batch_norm: bool,
                 item_batch_norm: bool):
        super(UserDefineModel, self).__init__()

        self._user_input_layer = tf.keras.layers.DenseFeatures(feature_columns=user_column_features)
        self._item_input_layer = tf.keras.layers.DenseFeatures(feature_columns=item_column_features)

        self._user_hidden_layers = []
        self._user_dropout_layers = []
        self._user_batch_norm_layers = []
        for hidden_unit in user_hidden_units:
            self._user_hidden_layers.append(tf.keras.layers.Dense(units=hidden_unit, activation=user_activation))
            if user_dropout:
                self._user_dropout_layers.append(tf.keras.layers.AlphaDropout(rate=user_dropout))
            if user_batch_norm:
                self._user_batch_norm_layers.append(tf.keras.layers.BatchNormalization())

        self._item_hidden_layers = []
        self._item_dropout_layers = []
        self._item_batch_norm_layers = []
        for hidden_unit in item_hidden_units:
            self._item_hidden_layers.append(tf.keras.layers.Dense(units=hidden_unit, activation=item_activation))
            if item_dropout:
                self._item_dropout_layers.append(tf.keras.layers.AlphaDropout(rate=item_dropout))
            if item_batch_norm:
                self._item_batch_norm_layers.append(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, mask=None):
        user_net = self._user_input_layer(inputs)
        item_net = self._item_input_layer(inputs)

        for index, hidden_layer in enumerate(self._user_hidden_layers):
            user_net = hidden_layer(user_net)
            if training:
                user_net = self._user_dropout_layers[index](user_net)
            user_net = self._user_batch_norm_layers[index](user_net)

        for index, hidden_layer in enumerate(self._item_hidden_layers):
            item_net = hidden_layer(item_net)
            if training:
                item_net = self._item_dropout_layers[index](item_net)
            item_net = self._item_batch_norm_layers[index](item_net)

        user_net = tf.math.l2_normalize(user_net)
        item_net = tf.math.l2_normalize(item_net)

        logits = tf.reduce_sum(tf.multiply(user_net, item_net), axis=1, keepdims=True)

        return logits
```

# 2. 自定义Estimator的基本架构

```python
class CommonEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model: tf.keras.Model,
                 head: tf.estimator.Head,
                 optimizer: tf.keras.optimizers.Optimizer,
                 config: tf.estimator.RunConfig,
                 params: dict = None,
                 warm_start_from: str = None):
        def model_fn(features, labels, mode):
            logits = model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
            trainable_variables = model.trainable_variables
            update_ops = model.updates
            # In TRAIN mode, create optimizer and assign global_step variable to
            # optimizer.iterations to make global_step increased correctly, as Hooks
            # relies on global step as step counter.
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer.iterations = training_util.get_or_create_global_step()
            return head.create_estimator_spec(features=features,
                                              mode=mode,
                                              logits=logits,
                                              labels=labels,
                                              optimizer=optimizer,
                                              trainable_variables=trainable_variables,
                                              train_op_fn=None,  # 自定义训练op。未设置optimizer时，该配置生效
                                              update_ops=update_ops,
                                              regularization_losses=None)

        super(CommonEstimator, self).__init__(model_fn=model_fn, params=params, config=config,
                                              warm_start_from=warm_start_from)
```

# 3. 自定义Head的基本架构

```python
class UserDefineHead(tf.estimator.Head):

    def __init__(self, task_type: str,
                 training_hooks: list = None,
                 training_chief_hooks: list = None,
                 evaluation_hooks: list = None,
                 prediction_hooks: list = None,
                 scaffold: tf.compat.v1.train.Scaffold = None):
        """

        :param task_type: 目前只支持classification或者regression两种取值
        :param training_hooks: 配置训练过程中的hooks，这个也可以配置在TrainSpec中，最终会将两者合并
        :param training_chief_hooks: 配置训练过程中只在chief节点执行的hooks
        :param evaluation_hooks: 配置评估过程中的hooks，这个也可以配置在EvalSpec中，最终会将两者合并
        :param prediction_hooks: 配置预测过程中的hooks
        :param scaffold: 这个还是不懂！！！！！！！！！！！！！！！！！！！
        """
        self._task_type = task_type
        self._training_hooks = training_hooks
        self._training_chief_hooks = training_chief_hooks
        self._evaluation_hooks = evaluation_hooks
        self._prediction_hooks = prediction_hooks
        self._scaffold = scaffold

    @property
    def name(self):
        return "UserDefineHead"

    @property
    def logits_dimension(self):
        return 1

    @property
    def loss_reduction(self):
        return tf.losses.Reduction.SUM_OVER_BATCH_SIZE

    def loss(self, labels, logits, features=None, mode=None, regularization_losses=None):
        return []

    def predictions(self, logits, keys=None):
        # 具体的预测指标，完全根据算法的评价指标来定。PredictionKeys只是提供了常用的评价指标
        prediction = dict()
        prediction[PredictionKeys.LOGITS] = logits
        prediction[PredictionKeys.LOGISTIC] = tf.math.sigmoid(logits)
        two_class_logits = tf.concat((tf.zeros_like(logits), logits), axis=-1)
        prediction[PredictionKeys.PROBABILITIES] = tf.math.softmax(two_class_logits)
        prediction[PredictionKeys.CLASS_IDS] = tf.expand_dims(tf.math.argmax(two_class_logits, axis=-1), axis=-1)
        return prediction

    def metrics(self, regularization_losses=None):
        pass

    def update_metrics(self, eval_metrics, features, logits, labels, mode=None, regularization_losses=None):
        pass

    def create_estimator_spec(self, features, mode, logits, labels=None, optimizer=None, trainable_variables=None,
                              train_op_fn=None, update_ops=None, regularization_losses=None):
        """
        :param features: 模型输入数据的特征
        :param mode: 模型执行的类型，包括训练、评估、预测三种
        :param logits: 模型call函数的返回值
        :param labels: 模型输入数据的标签
        :param optimizer: 模型优化器
        :param trainable_variables: 模型需要训练的参数集合，从model.trainable_variables获得
        :param train_op_fn: 在不指定optimizer的情况下，自定义模型训练参数的操作函数
        :param update_ops: 针对类似batchNorm层需要更新的操作，从model.updates获得
        :param regularization_losses: 额外需要加到模型loss里的列表，例如正则化loss
        :return:
        """
        # used during predict, evaluate and train
        predictions = self.predictions(logits)
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = dict()
            export_outputs["predict"] = tf.estimator.export.PredictOutput(predictions)
            if self._task_type == "classification":
                classification_output = tf.estimator.export.ClassificationOutput(
                    scores=predictions[PredictionKeys.PROBABILITIES], classes=None)
                export_outputs["serving_default"] = classification_output
                export_outputs["classification"] = classification_output
            elif self._task_type == "regression":
                regression_output = tf.estimator.export.RegressionOutput(
                    value=predictions[PredictionKeys.PREDICTIONS])
                export_outputs["serving_default"] = regression_output
                export_outputs["regression"] = regression_output
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                              predictions=predictions,
                                              export_outputs=export_outputs,
                                              prediction_hooks=self._prediction_hooks)

        # used during evaluate and train
        regularized_training_loss = self.loss(labels=labels, logits=logits, features=features, mode=mode,
                                              regularization_losses=regularization_losses)
        if mode == tf.estimator.ModeKeys.EVAL:
            # used during evaluate
            eval_metrics = self.metrics(regularization_losses=regularization_losses)
            eval_metric_ops = self.update_metrics(eval_metrics=eval_metrics,
                                                  features=features,
                                                  logits=logits,
                                                  labels=labels,
                                                  mode=mode,
                                                  regularization_losses=regularization_losses)
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                              predictions=predictions,
                                              loss=regularized_training_loss,
                                              eval_metric_ops=eval_metric_ops,
                                              evaluation_hooks=self._evaluation_hooks)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = base_head.create_estimator_spec_train_op(
                head_name=self.name,
                optimizer=optimizer,
                trainable_variables=trainable_variables,
                train_op_fn=train_op_fn,
                update_ops=update_ops,
                regularized_training_loss=regularized_training_loss,
                loss_reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)  # used during train
            return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                              loss=regularized_training_loss,
                                              predictions=predictions,
                                              train_op=train_op,
                                              training_hooks=self._training_hooks,
                                              training_chief_hooks=self._training_chief_hooks,  # 分布式训练只在chief生效
                                              scaffold=self._scaffold)
```

# 4. 模型参数解析的基本架构

```python
flags.DEFINE_string(name="model_name", default="testModelName", help="指定模型名称，导出SavedModel时使用")
flags.DEFINE_string(name="model_dir", default=None, help="指定模型的路径地址")
flags.DEFINE_string(name="keywords_file", default=None, help="指定关键词列表文件")
flags.DEFINE_string(name="keywords_weights_file", default=None, help="指定关键词向量文件目录")
flags.DEFINE_string(name="user_hidden_units", default="512,256,128", help="设置用户侧的隐含层")
flags.DEFINE_string(name="item_hidden_units", default="512,256,128", help="设置商品侧的隐含层")
flags.DEFINE_float(name="user_dropout", default=0.7, help="设置用户侧dropout")
flags.DEFINE_float(name="item_dropout", default=0.7, help="设置商品侧dropout")
flags.DEFINE_boolean(name="user_batch_norm", default=True, help="设置用户侧是否做batchNorm")
flags.DEFINE_boolean(name="item_batch_norm", default=True, help="设置商品侧是否做batchNorm")
flags.DEFINE_float(name="learning_rate", default=0.01, help="设置训练的学习率")

flags.DEFINE_string(name="train_files", default=None, help="训练数据文件列表，逗号分割")
flags.DEFINE_integer(name="train_batch_size", default=100, help="训练数据批次大小")
flags.DEFINE_integer(name="train_num_epochs", default=None, help="整个训练数据集重复次数")
flags.DEFINE_integer(name="train_shuffle_buffer_size", default=10000, help="训练数据混洗缓冲区大小")
flags.DEFINE_integer(name="train_reader_num_threads", default=1, help="训练数据读取并发线程数")
flags.DEFINE_integer(name="train_parser_num_threads", default=1, help="训练数据解析并发线程数")
flags.DEFINE_integer(name="max_train_steps", default=1000,
                     help="指定模型训练的最大步数。None表示步数为无穷大。不过最终的终止条件还需要看input_fn是否还有返回结果")

flags.DEFINE_string(name="eval_files", default=None, help="评估数据文件列表，逗号分割")
flags.DEFINE_integer(name="eval_batch_size", default=100, help="评估数据批次大小")
flags.DEFINE_integer(name="eval_reader_num_threads", default=1, help="评估数据读取并发线程数")
flags.DEFINE_integer(name="eval_parser_num_threads", default=1, help="评估数据解析并发线程数")
flags.DEFINE_integer(name="max_eval_steps", default=None,
                     help="指定最大的评估步数。None表示步数为无穷大。不过最终的终止条件还需要看input_fn是否还有返回结果")

flags.DEFINE_integer(name="start_delay_secs", default=120, help="第一次评估开始于start_delay_secs + throttle_secs")
flags.DEFINE_integer(name="throttle_secs", default=600, help="两次评估开始时间的间隔。如果没有新的checkpoints，则不进行评估。")

flags.DEFINE_string(name="predict_files", default=None, help="待预测数据文件列表，逗号分割")
flags.DEFINE_integer(name="predict_batch_size", default=2, help="待预测数据批次大小")
flags.DEFINE_integer(name="predict_reader_num_threads", default=1, help="待预测数据读取并发线程数")
flags.DEFINE_integer(name="predict_parser_num_threads", default=1, help="待预测数据解析并发线程数")

FLAGS = flags.FLAGS
```

# 5. Estimator模型导出的三种方法

```python
# 模型训练结束导出最后一份模型
def model_final_exporter(model_name, feature_spec):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return tf.estimator.FinalExporter(name=model_name,
                                      serving_input_receiver_fn=serving_input_receiver_fn)

# 评估效果超过之前所有已存在的模型效果，就导出模型
def model_best_exporter(model_name, feature_spec):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    def _auc_bigger(best_eval_result, current_eval_result):
        default_key = "auc"
        if not best_eval_result or default_key not in best_eval_result:
            raise ValueError('best_eval_result cannot be empty or no loss is found in it.')

        if not current_eval_result or default_key not in current_eval_result:
            raise ValueError('current_eval_result cannot be empty or no loss is found in it.')

        return best_eval_result[default_key] < current_eval_result[default_key]

    return tf.estimator.BestExporter(name=model_name,
                                     serving_input_receiver_fn=serving_input_receiver_fn,
                                     compare_fn=_auc_bigger,
                                     exports_to_keep=1)

# 每次评估都导出模型，默认最多保存5份
def model_latest_exporter(model_name, feature_spec):
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return tf.estimator.LatestExporter(name=model_name,
                                       exports_to_keep=3,
                                       serving_input_receiver_fn=serving_input_receiver_fn)
```

# 6. Estimator模型调用的基本架构

```python
def input_fn(files, batch_size, features, num_epochs,
             shuffle_buffer_size=None, reader_num_threads=1, parser_num_threads=2):
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=files, batch_size=batch_size, features=features, label_key="label", num_epochs=num_epochs,
        shuffle=bool(shuffle_buffer_size), shuffle_buffer_size=shuffle_buffer_size,
        reader_num_threads=reader_num_threads, parser_num_threads=parser_num_threads, drop_final_batch=True)

def main(_):
    # config = tf.estimator.RunConfig(
    #     model_dir=FLAGS.model_dir,
    #     tf_random_seed=None,
    #     save_summary_steps=100,
    #     save_checkpoints_steps=object(),
    #     save_checkpoints_secs=object(),
    #     session_config=ConfigProto(),
    #     keep_checkpoint_max=5,
    #     keep_checkpoint_every_n_hours=10000,
    #     log_step_count_steps=100,
    #     train_distribute=None,
    #     device_fn=None,
    #     protocol=None,
    #     eval_distribute=None,
    #     experimental_distribute=None,
    #     experimental_max_worker_delay_secs=None,
    #     session_creation_timeout_secs=7200)
    config = None
    params = {}
    warm_start_from = None  # tf.estimator.WarmStartSettings()
    user_column_features, item_column_features = get_column_features(keywords_file=FLAGS.keywords_file,
                                                                     keywords_weights_file=FLAGS.keywords_weights_file,
                                                                     eval_date=datetime.datetime.now())
    head = tf.estimator.BinaryClassHead()
    model = UserDefineModel()
    estimator = EstimatorFactory(model_dir=FLAGS.model_dir,
                                 model=model,
                                 head=head,
                                 optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
                                 config=config,
                                 params=params,
                                 warm_start_from=warm_start_from)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            files=FLAGS.train_files.split(","),
            batch_size=FLAGS.train_batch_size,
            features=column_schema,
            num_epochs=FLAGS.train_num_epochs,
            shuffle_buffer_size=FLAGS.train_shuffle_buffer_size,
            reader_num_threads=FLAGS.train_reader_num_threads,
            parser_num_threads=FLAGS.train_parser_num_threads),
        max_steps=FLAGS.max_train_steps,
        hooks=[
            # tf.estimator.SessionRunHook(),
            # tf.estimator.CheckpointSaverHook(),
            # tf.estimator.FeedFnHook(),
            # tf.estimator.FinalOpsHook(),
            # tf.estimator.GlobalStepWaiterHook(),
            # tf.estimator.LoggingTensorHook(),
            # tf.estimator.NanTensorHook(),
            # tf.estimator.ProfilerHook(),
            # tf.estimator.StepCounterHook(),
            # tf.estimator.StopAtStepHook(),
            # tf.estimator.SummarySaverHook()
        ])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            files=FLAGS.eval_files.split(","),
            batch_size=FLAGS.eval_batch_size,
            features=column_schema,
            num_epochs=1,
            shuffle_buffer_size=None,
            reader_num_threads=FLAGS.eval_reader_num_threads,
            parser_num_threads=FLAGS.eval_parser_num_threads),
        steps=FLAGS.max_eval_steps,
        name="eval",  # 评估结果文件保存的目录名称
        hooks=[],
        exporters=[
            model_best_exporter(model_name=FLAGS.model_name + "1", feature_spec=column_schema),
            model_final_exporter(model_name=FLAGS.model_name + "2", feature_spec=column_schema),
            model_latest_exporter(model_name=FLAGS.model_name + "3", feature_spec=column_schema)],
        start_delay_secs=FLAGS.start_delay_secs,
        throttle_secs=FLAGS.throttle_secs)

    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    # estimator.train(
    #     input_fn=lambda: input_fn(
    #         files=FLAGS.train_files.split(","),
    #         batch_size=FLAGS.train_batch_size,
    #         features=feature_spec,
    #         num_epochs=FLAGS.train_num_epochs,
    #         shuffle_buffer_size=FLAGS.train_shuffle_buffer_size,
    #         reader_num_threads=FLAGS.train_reader_num_threads,
    #         parser_num_threads=FLAGS.train_parser_num_threads),
    #     hooks=[],
    #     steps=100,  # 每次调度train，执行的最大步数，不能和max_steps同时设置
    #     max_steps=None,  # 每次调度train，会从checkpoint中检查已训练步数，接着进行步数计数，不能和steps同时设置
    #     saving_listeners=[])
    #
    # metric_dict = estimator.evaluate(
    #     input_fn=lambda: input_fn(
    #         files=FLAGS.eval_files.split(","),
    #         batch_size=FLAGS.eval_batch_size,
    #         features=feature_spec,
    #         num_epochs=1,
    #         shuffle_buffer_size=None,
    #         reader_num_threads=FLAGS.eval_reader_num_threads,
    #         parser_num_threads=FLAGS.eval_parser_num_threads),
    #     steps=FLAGS.max_eval_steps,
    #     hooks=[],
    #     checkpoint_path=None,
    #     name="eval2")

    predict_dict_iterator = estimator.predict(
        input_fn=lambda: input_fn(
            files=FLAGS.predict_files.split(","),
            batch_size=FLAGS.predict_batch_size,
            features=column_schema,
            num_epochs=1,
            shuffle_buffer_size=None,
            reader_num_threads=FLAGS.predict_reader_num_threads,
            parser_num_threads=FLAGS.predict_parser_num_threads),
        predict_keys=None,  # [PredictionKeys.PROBABILITIES, PredictionKeys.LOGISTIC],  # 指定输出的预测指标，None表示全部输出
        hooks=[],
        checkpoint_path=None,  # 如果为None，则使用estimator对应的model_dir中最新的checkpoint
        yield_single_examples=False)  # 是否将批次拆分为单元素返回

    count = 0
    for predict_dict in predict_dict_iterator:
        count += 1
        print(predict_dict)
        if count > 100:
            break


if __name__ == '__main__':
    app.run(main=main, argv=None)
```
