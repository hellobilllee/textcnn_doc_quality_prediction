#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import shutil
from data_helper import  read_vocab,process_train_raw_file, process_file
from preprocess import generate_w2v
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from cnn_model import TCNNConfig, TextCNN

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
def batch_iter(x_title,x_content,x_keyword,x_auxilary, y, batch_size=256):
    """生成批次数据"""
    data_len = len(y)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_title_shuffle = x_title[indices]
    x_content_shuffle = x_content[indices]
    x_keyword_shuffle = x_keyword[indices]
    x_auxilary_shuffle = x_auxilary[indices]
    y_shuffle = y[indices]
    print(num_batch)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_title_shuffle[start_id:end_id],x_content_shuffle[start_id:end_id],x_keyword_shuffle[start_id:end_id],x_auxilary_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
def feed_data(x_title_batch,x_content_batch,x_keyword_batch,x_auxilary_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x_title: x_title_batch,
        model.input_x_content: x_content_batch,
        model.input_x_keyword: x_keyword_batch,
        model.input_x_auxilary: x_auxilary_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict
def evaluate(sess, title_x, content_x, keyword_x, auxilary_x, y):
    """评估在某一数据上的准确率和损失"""
    data_len = len(y)
    batch_eval = batch_iter(title_x, content_x, keyword_x, auxilary_x, y, 128)
    total_loss = 0.0
    total_acc = 0.0
    # total_auc = 0.0
    for title_x_batch, content_x_batch, keyword_x_batch, auxilary_x_batch, y_batch in batch_eval:
        batch_len = len(y_batch)
        feed_dict = feed_data(title_x_batch, content_x_batch, keyword_x_batch,auxilary_x_batch,  y_batch, 1.0)
        loss, acc= sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
        # total_auc += auc * batch_len
    return total_loss / data_len, total_acc / data_len
def train(filename=None):
    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    if filename is not None:
        df = process_train_raw_file(filename)
    elif os.path.exists("data/train_test_files/qttnews.train.csv"):
        df = pd.read_csv("data/train_test_files/qttnews.train.csv",header=0,index_col=None)
    elif os.path.exists("data/raw_data/model_data_2019-1-23_data2.txt"):
        df = process_train_raw_file("data/raw_data/model_data_2019-1-23_data2.txt")
    title_pad, content_pad, keyword_pad, auxilary, y_pad = process_file(df, word_to_id, title_max_length=20, content_max_length=6000, keyword_max_length=8)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_pad)))
    title_pad, content_pad, keyword_pad, auxilary, y_pad = title_pad[shuffle_indices], content_pad[shuffle_indices], keyword_pad[shuffle_indices], auxilary[shuffle_indices], y_pad[shuffle_indices]
    # Split train/val set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = int(dev_sample_percentage * float(len(y_pad)))
    title_train, content_train, keyword_train, auxilary_train, y_train = title_pad[dev_sample_index:], content_pad[dev_sample_index:], keyword_pad[dev_sample_index:], auxilary[dev_sample_index:], y_pad[dev_sample_index:]
    title_val, content_val, keyword_val, auxilary_val, y_val = title_pad[:dev_sample_index], content_pad[:dev_sample_index], keyword_pad[:dev_sample_index], auxilary[:dev_sample_index], y_pad[:dev_sample_index]
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print("begin the training")
    # Training
    # ==================================================

    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    else:
        for the_file in os.listdir(tensorboard_dir):
            file_path = os.path.join(tensorboard_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    # tf.summary.scalar("auc", model.auc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # 创建session

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # session.run(tf.initialize_local_variables())
    writer.add_graph(sess.graph)
    if tf.train.checkpoint_exists(save_path):
        print("restoring the model")
        saver.restore(sess=sess, save_path=save_path)  # 读取保存的模型

    # if not config.disable_word_embeddings:
    #     if os.path.exists(word_vector_dir):
    #         initW = pd.read_csv(word_vector_dir,header=None,index_col=None).values
    #         sess.run(model.embedding.assign(initW))

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    # best_auc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(title_train, content_train, keyword_train, auxilary_train, y_train, config.batch_size)
        for x_title_batch,x_content_batch,x_keyword_batch,x_auxilary_batch, y_batch in batch_train:
            feed_dict = feed_data(x_title_batch,x_content_batch,x_keyword_batch,x_auxilary_batch,y_batch, config.dropout_keep_prob)

            _, global_step, train_summaries, train_loss, train_accuracy = sess.run([model.optim, model.global_step,
                                                                                    merged_summary, model.loss,
                                                                                    model.acc], feed_dict=feed_dict)
            if global_step % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, global_step)

            if global_step % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train= sess.run([model.loss, model.acc], feed_dict=feed_dict)

                loss_val, acc_val= evaluate(sess,title_val, content_val, keyword_val, auxilary_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = global_step
                    saver.save(sess=sess, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                # print("Time usage:", time_dif)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val,acc_val, time_dif, improved_str))

            # sess.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1
            # print(total_batch)
            if global_step - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环

        if flag:  # 同上
            break
        config.lr *= config.lr_decay

def test():
    print("Loading test data...")
    start_time = time.time()
    df = pd.read_csv(test_dir,header=0,index_col=None)
    x_title_test, x_content_test, x_keyword_test, x_auxilary_test, y_test = process_file(df, word_to_id,config.title_seq_length,config.content_seq_length,config.keyword_seq_length)

    # x_test, y_test = process_file_v1(test_dir, word_to_id, config.seq_length)
    # Testing
    # ==================================================
    sess= tf.Session()
    sess.run(tf.global_variables_initializer())
    # session.run(tf.initialize_local_variables())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(sess, x_title_test, x_content_test, x_keyword_test, x_auxilary_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = config.batch_size
    data_len = len(y_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(y_test), dtype=np.int32)  # 保存预测结果
    y_pred_prob = np.zeros(shape=len(y_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x_title: x_title_test[start_id:end_id],
            model.input_x_content: x_content_test[start_id:end_id],
            model.input_x_keyword: x_keyword_test[start_id:end_id],
            model.input_x_auxilary: x_auxilary_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = sess.run(model.y_pred_cls, feed_dict=feed_dict)
        y_pred_prob[start_id:end_id] = sess.run(model.y_pred_prob, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=["low_quality","high_quality"]))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
    # 评估auc
    print("auc...")
    print(roc_auc_score(y_test_cls, y_pred_prob) if len(y_test_cls) > 0 else 0)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

if __name__ == '__main__':
    if len(sys.argv)>3 or len(sys.argv)<2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test] [train/test file]""")
    base_dir = 'data/train_test_files'
    train_dir = os.path.join(base_dir, 'qttnews.train.csv')
    dev_sample_percentage = 0.1
    test_dir = os.path.join(base_dir, 'qttnews.test.csv')
    vocab_dir = "data/w2v/qttnews.vocab.txt"
    word_vector_dir = "data/w2v/qttnews.vector.txt"
    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
    # tf.reset_default_graph()
    print('Configuring CNN model...')
    config = TCNNConfig()

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        print("no vocabulary file, need to generate it ")
        generate_w2v()
    # categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    # trans vector file to numpy file
    if not os.path.exists(word_vector_dir):
        print("no pretrained w2v exists, generate the w2v")
        generate_w2v()
    else:
        print("load w2v embeddings")
        config.pre_trianing = pd.read_csv(word_vector_dir, header=None, index_col=None).values
    model = TextCNN(config)

    if sys.argv[1] == 'train' and len(sys.argv)==3:
        train(filename=sys.argv[2])
    elif sys.argv[1] == 'train' and len(sys.argv)==2:
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        raise ValueError("""usage: python run_cnn.py [train / test] [train/test file]""")