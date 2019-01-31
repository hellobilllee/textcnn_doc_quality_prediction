# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
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
    learning_rate = 1e-3  # 学习率

    batch_size = 256 # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 500  # 每多少轮存入tensorboard
    disable_word_embeddings = False

class TextCNN(object):
    """文本分类，CNN模型"""
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, config):

        self.config = config
        # 三个待输入的数据
        self.input_x_title = tf.placeholder(tf.int32, [None, self.config.title_seq_length], name='input_x_title')
        self.input_x_content = tf.placeholder(tf.int32, [None, self.config.content_seq_length], name='input_x_content')
        self.input_x_keyword = tf.placeholder(tf.int32, [None, self.config.keyword_seq_length], name='input_x_keyword')
        self.input_x_auxilary = tf.placeholder(tf.float32, [None, self.config.auxilary], name='auxilary')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            self.embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0), name='word_embedding',trainable=self.config.disable_word_embeddings)
            # self.embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0), name='word_embedding',trainable=True)
            embedding_inputs_title = tf.nn.embedding_lookup(self.embedding, self.input_x_title)
            embedding_inputs_content = tf.nn.embedding_lookup(self.embedding, self.input_x_content)
            embedding_inputs_keyword = tf.nn.embedding_lookup(self.embedding, self.input_x_keyword)
            # embedding_inputs_title_expanded = tf.expand_dims(embedding_inputs_title, -1)
            embedding_inputs_content_expanded = tf.expand_dims(embedding_inputs_content, -1)
            # embedding_inputs_keyword_expanded = tf.expand_dims(embedding_inputs_keyword, -1)
        # print("embedding_inputs_title:",embedding_inputs_title.get_shape())
        # print("embedding_inputs_content:",embedding_inputs_content.get_shape())
        # print("embedding_inputs_keyword:",embedding_inputs_keyword.get_shape())
        # print("embedding_inputs_content_expanded :",embedding_inputs_content_expanded.get_shape())

        # Create a convolution + maxpool layer for each filter size of the content
        pooled_outputs_content = []
        for i, filter_size in enumerate(self.config.content_filter_sizes):
            with tf.name_scope('content-conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.content_num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_content_%s'%i)
                b = tf.Variable(tf.constant(0.1, shape=[self.config.content_num_filters]), name='b_content_%s'%i)
                conv = tf.nn.conv2d(
                    embedding_inputs_content_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.content_seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs_content.append(pooled)

        # Combine all the pooled features
        num_filters_total_content = self.config.content_num_filters * len(self.config.content_filter_sizes)
        h_pool_content = tf.concat(pooled_outputs_content, 3)
        h_pool_flat_content = tf.reshape(h_pool_content, [-1, num_filters_total_content])
        # print("h_pool_flat_content :", h_pool_flat_content .get_shape())
        with tf.name_scope("title-conv-maxpool"):
            # CNN layer
            conv_title = tf.layers.conv1d(embedding_inputs_title, self.config.title_num_filters, self.config.title_filter_sizes, name='conv_title',reuse=tf.AUTO_REUSE)
            # global max pooling layer
            h_pool_flat_title = tf.reduce_max(conv_title, reduction_indices=[1], name='gmp_title')
            # print("h_pool_flat_title:",h_pool_flat_title.get_shape())
        with tf.name_scope("keyword-conv-maxpool"):
            # CNN layer
            conv_keyword = tf.layers.conv1d(embedding_inputs_keyword, self.config.keyword_num_filters, self.config.keyword_filter_sizes, name='conv_keyword',reuse=tf.AUTO_REUSE)
            # global max pooling layer
            h_pool_flat_keyword = tf.reduce_max(conv_keyword, reduction_indices=[1], name='gmp_keyword')
            # print("h_pool_flat_keyword:",h_pool_flat_keyword.get_shape())

        # print("self.input_x_auxilary:",self.input_x_auxilary.get_shape())

        concat_input = tf.concat([h_pool_flat_content,h_pool_flat_title, h_pool_flat_keyword,self.input_x_auxilary], axis=1)
        # print("concat_input:",concat_input.get_shape())
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(concat_input, self.config.hidden_dim, name='fc1',reuse=tf.AUTO_REUSE)
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # print("fc:", fc.get_shape())
            # 分类器
            self.logits = tf.layers.dense(concat_input, self.config.num_classes, name='fc2',reuse=tf.AUTO_REUSE)
            self.y_pred_prob= tf.nn.softmax(self.logits)[:, 1]
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # with tf.name_scope("auc"):
        #     auc
            # self.auc = tf.metrics.auc(tf.argmax(self.input_y, 1), self.y_pred_prob)
            # self.auc = roc_auc_score(tf.argmax(self.input_y, 1), self.y_pred_prob)