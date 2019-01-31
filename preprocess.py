import os
import pandas as pd
import multiprocessing
from gensim.models import Word2Vec
from data_helper import get_content_from_htmlpage, word_cutter


def generate_w2v():

    if os.path.exists("data/news_content/data1.txt"):
        files = os.listdir("data/news_content/")
        wc = word_cutter("data/dict")
        df_files = pd.Series()
        for file in files:
            df_file_temp = pd.read_csv("data/news_content/"+file,sep="                    dfdgasddfasdfad",header=None,index_col=None,encoding='utf-8').loc[9:,0].apply(get_content_from_htmlpage).apply(lambda x:wc.cut_words(x,return_list=True))
            print("data/news_content/"+file,len(df_file_temp))
            df_files = df_files.append(df_file_temp)
    elif os.path.exists("data/train_test_files/qttnews.train.txt"):
        print("no news content directory, get content from train files")
        df_files = pd.read_csv("data/train_test_files/qttnews.train.txt",header=0,index_col=None)
        df_files = df_files["detail"]
        df_files = df_files.apply(lambda x: x.split(" "))
    elif os.path.exists("data/raw_data/model_data_2019-1-23_data2.txt"):
        df_files = pd.read_csv("data/raw_data/model_data_2019-1-23_data2.txt", sep="\t", header=None, index_col=None)
        df_files.columns = ['content_id', 'title', 'detail', 'type', 'rank', 'keywords', 'rec_validread', 'rec_validshow',
                      "create_time", "update_time", "status"]
        type_cat = [16, 1, 4, 42, 9, 7, 8, 6, 17, 19, 25, 5, 12, 13, 23, 10, 27, 15, 14, 18]
        rank_cat = [1, 2, 3, 4, 5]
        df_files = df_files[df_files['type'].isin(type_cat)]
        df_files = df_files[df_files['rank'].isin(rank_cat)]["detail"]
        df_files = df_files.apply(get_content_from_htmlpage).apply(lambda x:wc.cut_words(x,return_list=True))

    # df_files = df_files.apply(lambda x: x.split(" "))

    model = Word2Vec(df_files, size = 128, window =5, min_count = 5, workers = multiprocessing.cpu_count(),iter=10)
    if not os.path.exists("data/w2v/"):
        os.makedirs("data/w2v/")
    model.save("data/w2v/qttnews_w2c.model")
    model.wv.save_word2vec_format("data/w2v/qttnews_w2c.kv", binary=False)
    open("data/w2v/qttnews.vocab.txt", mode='w', encoding='utf-8').write('\n'.join(model.wv.index2word) + '\n')
    # pd.Series(model.wv.index2word).to_csv("data/w2v/qttnews.vocab.txt", header=False, index=False)
    pd.DataFrame(model.wv.vectors).to_csv("data/w2v/qttnews.vector.txt", header=False, index=False)


