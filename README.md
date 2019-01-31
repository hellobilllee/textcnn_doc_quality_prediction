# Text Classification with CNN and RNN

使用卷积神经网络以及循环神经网络进行中文文本分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

本文是基于TensorFlow在中文数据集上的简化实现，使用了预训练的w2v对中文文本进行分类，达到了较好的效果。

文中所使用的Conv1D与论文中有些不同，详细参考官方文档：[tf.nn.conv1d](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d)

## 环境

- Python 2/3 
- TensorFlow 1.3以上
- numpy
- scikit-learn
- scipy
- jieba
- gensim

## 数据集

使用十多万用于推送的语料进行训练与测试，数据特征包括新闻的title, detail, keyword, rank, type, publish_time；

模型为二分类模型，标签通过综合考量文章pv、ctr(valid_pv/valid_display)、status（审核状态）（因为都是推荐的语料，历史语料有status审核标签）生成；

数据集经清洗后共106345条，用1%做验证数据，测试数据为线上实时拉取；

用于生成为词典和w2v向量的数据为60多万条历史推荐数据


## 预处理

`preprocess.py`及`data_helper.py`为数据的预处理文件。
- `generate_w2v()`: 生成为词汇表和w2v;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |

## CNN卷积神经网络

### 配置项

CNN可配置的参数如下所示，在`cnn_model.py`中。

```python
class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 128  # 词向量维度
    title_seq_length = 20  # title序列长度
    content_seq_length = 6000  # content序列长度
    keyword_seq_length = 8  # keyword序列长度
    auxilary= 30  # 辅助信息序列长度，type:one_hot,rank:one-hot,weekday,holiday,'title_length', 'title_token_length','detail_length', 'detail_token_length'
    num_classes = 2  # 类别数
    title_filter_sizes = 5
    content_filter_sizes = [3,4,5]
    keyword_filter_sizes = 5
    title_num_filters = 64  # 卷积核数目
    content_num_filters= 128  # 卷积核数目
    keyword_num_filters= 64  # 卷积核数目
    vocab_size = 510831 # 词汇表大小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    lr = 1e-3  # 学习率
    lr_decay = 0.9          #learning rate decay
    clip = 5.0              #gradient clipping threshold

    batch_size = 256 # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 500  # 每多少轮存入tensorboard
    # disable_word_embeddings = False
    pre_trianing = None   #use vector_char trained by word2vec
```

### CNN模型

具体参看`cnn_model.py`的实现。

大致结构如下：

![images/cnn_architecture](images/cnn_architecture.png)

### 训练与验证

运行 `python run_cnn.py train`，可以开始训练。

如果想指定训练文件,则将文件作为第三个参数传入 `python run_cnn.py train train_file`，可以开始训练。
```
Configuring CNN model...
Configuring TensorBoard and Saver...
Loading training and validation data...
Time usage: 0:00:14
Training and evaluating...
Epoch: 1
Iter:      0, Train Loss:    2.3, Train Acc:  10.94%, Val Loss:    2.3, Val Acc:   8.92%, Time: 0:00:01 *
Iter:    100, Train Loss:   0.88, Train Acc:  73.44%, Val Loss:    1.2, Val Acc:  68.46%, Time: 0:00:04 *
Iter:    200, Train Loss:   0.38, Train Acc:  92.19%, Val Loss:   0.75, Val Acc:  77.32%, Time: 0:00:07 *
Iter:    300, Train Loss:   0.22, Train Acc:  92.19%, Val Loss:   0.46, Val Acc:  87.08%, Time: 0:00:09 *
Iter:    400, Train Loss:   0.24, Train Acc:  90.62%, Val Loss:    0.4, Val Acc:  88.62%, Time: 0:00:12 *
Iter:    500, Train Loss:   0.16, Train Acc:  96.88%, Val Loss:   0.36, Val Acc:  90.38%, Time: 0:00:15 *
Iter:    600, Train Loss:  0.084, Train Acc:  96.88%, Val Loss:   0.35, Val Acc:  91.36%, Time: 0:00:17 *
Iter:    700, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:   0.26, Val Acc:  92.58%, Time: 0:00:20 *
Epoch: 2
Iter:    800, Train Loss:   0.07, Train Acc:  98.44%, Val Loss:   0.24, Val Acc:  94.12%, Time: 0:00:23 *
Iter:    900, Train Loss:  0.092, Train Acc:  96.88%, Val Loss:   0.27, Val Acc:  92.86%, Time: 0:00:25
Iter:   1000, Train Loss:   0.17, Train Acc:  95.31%, Val Loss:   0.28, Val Acc:  92.82%, Time: 0:00:28
Iter:   1100, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.23, Val Acc:  93.26%, Time: 0:00:31
Iter:   1200, Train Loss:  0.081, Train Acc:  98.44%, Val Loss:   0.25, Val Acc:  92.96%, Time: 0:00:33
Iter:   1300, Train Loss:  0.052, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  93.58%, Time: 0:00:36
Iter:   1400, Train Loss:    0.1, Train Acc:  95.31%, Val Loss:   0.22, Val Acc:  94.12%, Time: 0:00:39
Iter:   1500, Train Loss:   0.12, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  93.58%, Time: 0:00:41
Epoch: 3
Iter:   1600, Train Loss:    0.1, Train Acc:  96.88%, Val Loss:   0.26, Val Acc:  92.34%, Time: 0:00:44
Iter:   1700, Train Loss:  0.018, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  93.46%, Time: 0:00:47
Iter:   1800, Train Loss:  0.036, Train Acc: 100.00%, Val Loss:   0.28, Val Acc:  92.72%, Time: 0:00:50
No optimization for a long time, auto-stopping...
```

在验证集上的最佳效果为94.12%，且只经过了3轮迭代就已经停止。

准确率和误差如图所示：

![images](images/acc_loss.png)


### 测试

运行 `python run_cnn.py test` 在测试集上进行测试。

测试数据通过运行 `python test.py` 生成

```
Configuring CNN model...
Loading test data...
Testing...
Test Loss:   0.24, Test Acc:  90.54%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

 low_quality       0.91      0.98      0.94     16897
high_quality       0.88      0.63      0.73      4372

   micro avg       0.91      0.91      0.91     21269
   macro avg       0.89      0.80      0.84     21269
weighted avg       0.90      0.91      0.90     21269

Confusion Matrix...
[[16515   382]
 [ 1631  2741]]

Time usage: 0:12:05
```

在测试集上的准确率达到了90.04%，且各类的precision, recall和f1-score都超过了0.9。

从混淆矩阵也可以看出分类效果非常优秀。


## 预测

为方便预测，repo 中 `predict.py` 提供了 CNN 模型的预测方法。

9 参考
=
1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
2. [gaussic/text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
3. [YCG09/tf-text-classification](https://github.com/YCG09/tf-text-classification)


