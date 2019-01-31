import sys
from html.parser import HTMLParser
from importlib import reload
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr
import time
import traceback
import re
import jieba
import math
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

class word_cutter(object):
    def __init__(self, dict_path='data/dict'):
        """用户字典的路径"""
        # self.dict_path = os.path.dirname(os.path.realpath(os.getcwd())) + '/' + dict_path
        self.dict_path = dict_path

        self.stopwords = {}

        stop_word_dir = '%s/stopwords.txt' % self.dict_path
        user_dict_dir = '%s/tag_dict.dict' % self.dict_path

        # 加载停用词和字典
        #       if os.path.exists(stop_word_dir):
        self.load_stopwords('%s/stopwords.txt' % self.dict_path)
        #       if os.path.exists(user_dict_dir):
        jieba.load_userdict('%s/tag_dict.dict' % self.dict_path)

    def load_stopwords(self, stopwords_file):
        """
        加载停用词
        """
        step_1 = time.time()
        print('Loading Stop Words...')
        f = open(stopwords_file, 'r', encoding='UTF-8')
        line = f.readline()
        while line:
            try:
                word = line.strip()
                self.stopwords[word] = 1
            except:
                traceback.print_exc()
            line = f.readline()
        f.close()
        self.stopwords[u' '] = 1
        self.stopwords[u'\n'] = 1
        self.stopwords[u'\r'] = 1
        self.stopwords[u'\r\n'] = 1
        step_2 = time.time()
        print('Finish,using:%s' % (step_2 - step_1))

    def clean_html(self, raw_html):
        """
        去除html标签
        """
        cleanr = re.compile('<.*?>')
        #         re_html=re.compile('<[^>]+>')#从'<'开始匹配，不是'>'的字符都跳过，直到'>'
        re_punc = re.compile('[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*“”《》：（）]+')  # 去除标点符号
        re_digits_letter = re.compile('[a-zA-Z0-9]+')  # 去除数字及字母
        #         line=re.sub(re_html,' ',line)
        #         line=re.sub(re_punc," ",line)

        line = re.sub(cleanr, ' ', raw_html)
        line = re.sub(re_punc, "", line)
        cleantext = re.sub(re_digits_letter, "", line)
        #         cleantext = re.sub(cleanr, ' ', raw_html)
        return cleantext
    def cut_words(self,strs,min_word_length=2,return_list=False):
        """
        strs支持字符串
        """
        if return_list:
            return [x.strip() for x in jieba.cut(re.sub(re.compile('<.*?>'), ' ', strs)) if
             x.strip() and len(x.strip()) >=min_word_length and x not in self.stopwords]
        else:
            return " ".join([x.strip() for x in jieba.cut(re.sub(re.compile('<.*?>'), ' ', strs)) if
                         x.strip() and len(x.strip()) >=min_word_length and x not in self.stopwords])


class htmlparser(HTMLParser):
    # a_text = False
    # data_list = []
    def __init__(self):
        HTMLParser.__init__(self)
        self.links = []
        self.tag = None
        self.a_text = False
        # self.content = ""
        # print (self.a_text)

    def handle_starttag(self, tag, attr):
        if tag == 'p':
            # print("starttag: %s" %tag)
            self.a_text = True
            # print (dict(attr))
            # print (self.a_text)

    def handle_endtag(self, tag):
        if tag == 'p':
            # print("endtag: %s" %tag)
            self.a_text = False
            # print (self.a_text)

    def handle_data(self, data):
        if self.a_text:
            # print (self.a_text)
            # print (data)
            self.links.append(data)
            # self.content = "".join(self.links[:-1])
        # print (self.content)
        # data_list.append(data)

def get_content_from_htmlpage(htmlpage):
    yk = htmlparser()
    yk.feed(htmlpage)
    #     print (yk.links)
    content = "".join(yk.links[:-1])
    #     print (content)
    yk.close()
    return content

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content
def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
def read_category(type="label"):
    """读取分类目录，固定"""
    # categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    if type=="label":
        categories = [0,1]
    elif type == "rank":
        categories = [1, 2, 3, 4, 5]
    elif type == "type":
        categories = [16,  1,  4, 42,  9,  7,  8,  6, 17, 19, 25,  5, 12, 13, 23, 10, 27,15, 14, 18]
    elif type == "weekday":
        categories = [0,1, 2, 3, 4, 5,6]
    elif type == "holiday":
        categories = [0,1]
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id
def content_to_id(content,word_to_id):
    return [word_to_id[word] for word in content.split(" ") if word in word_to_id]
def content_to_id_kw(content,word_to_id):
    return [word_to_id[word] for word in content.split(",") if word in word_to_id]

def process_train_raw_file(train_file="data/raw_data/model_data_2019-1-23_data2.txt"):
    print("begin to process the raw file ")
    print("read the data:")
    df = pd.read_csv(train_file, sep="\t", header=None, index_col=None)
    df.columns = ['content_id', 'title', 'detail', 'type', 'rank', 'keywords', 'rec_validread', 'rec_validshow',
                  "create_time", "update_time", "status"]
    type_cat = [16,  1,  4, 42,  9,  7,  8,  6, 17, 19, 25,  5, 12, 13, 23, 10, 27,15, 14, 18]
    rank_cat = [1, 2, 3, 4, 5]
    df = df[df['type'].isin(type_cat)]
    df = df[df['rank'].isin(rank_cat)]

    print("get the label")
    df["ctr"] = df["rec_validread"] * 1.0 / (df["rec_validshow"] + 0.0000001)
    df["rec_validread_log"] = np.log10(df["rec_validread"] + 1)
    df["rec_validread_log"] = 1.0 * (df["rec_validread_log"] - df["rec_validread_log"].min()) / (
                df["rec_validread_log"].max() - df["rec_validread_log"].min())
    df = df[df.ctr <= 1]
    df["label"] = df["rec_validread_log"] * df["ctr"]
    df.loc[df['status'] == 11, "label"] = df.loc[df['status'] == 11, "label"].apply(lambda x: 0.8 * x)
    quantile_8 = df["label"].quantile(0.8)
    df['label'] = df['label'].apply(lambda x: 1 if x > quantile_8 else 0)

    df['create_time'] = pd.to_datetime(df['create_time'])
    df['update_time'] = pd.to_datetime(df['update_time'])
    df["publish_to_update_hour"] = (df['update_time'] - df['create_time']) / np.timedelta64(1, 'h')
    df = df[df["publish_to_update_hour"] <= 36]
    df["publish_to_update_hour"] = df["publish_to_update_hour"].apply(lambda x: 1.0 if x<=3 else math.log(x,3))

    print("begin to clean and cut the content ")
    wc = word_cutter("data/dict")
    df['detail'] = df['detail'].apply(get_content_from_htmlpage)
    df['title_length'] = df['title'].apply(lambda x: len(wc.clean_html(x)))
    df['title'] = df['title'].apply(lambda x: wc.cut_words(x))
    df["title_token_length"] = df['title'].apply(lambda x: len(x.split(" ")))
    df['detail_length'] = df['detail'].apply(lambda x: len(wc.clean_html(x)))
    df['detail'] = df['detail'].apply(lambda x: wc.cut_words(x))
    df['detail_token_length'] = df['detail'].apply(lambda x: len(x.split(" ")))
    df = df[((df['title_length'] > 1) & (df['title_token_length'] > 1)) & ((df['detail_length'] > 10)& (df['detail_token_length'] > 3))]
    df['detail_length'] = df['detail_length'].apply(lambda x: math.log2(x))

    df.drop_duplicates(inplace=True)
    df = df[~df.isnull().any(axis=1)]

    df = df[['content_id','label','detail','detail_length','detail_token_length', 'title','title_length',"title_token_length", 'keywords', 'type', 'rank',"publish_to_update_hour"]]
    print("save the prepared data:")
    print(df.describe())
    df.to_csv("data/train_test_files/qttnews.train.csv",header=True,index=False)

    return df

def process_file(df,word_to_id, title_max_length=20, content_max_length=6000, keyword_max_length=8,test=False):

    """将df转换为id表示"""
    for feat in ['title_length', 'title_token_length','detail_length', 'detail_token_length',"publish_to_update_hour"]:
        df[feat] = (df[feat]-df[feat].min())*1.0/(df[feat].max()-df[feat].min())
    auxilary = df[['title_length', 'title_token_length','detail_length', 'detail_token_length',"publish_to_update_hour"]].values
    for feat in ["rank","type"]:
        categories, cat_to_id = read_category(type=feat)
        # 将标签转换为one-hot表示
        auxilary = np.concatenate((auxilary,kr.utils.to_categorical(df[feat].apply(lambda x: cat_to_id[x]).tolist(), num_classes=len(categories))), axis=1)

    title_id = df['title'].apply(lambda x: content_to_id(str(x),word_to_id)).tolist()
    content_id = df['detail'].apply(lambda x: content_to_id(str(x),word_to_id)).tolist()
    keyword_id = df['keywords'].apply(lambda x: content_to_id_kw(str(x),word_to_id)).tolist()

    print("title_id:",title_id[:2])
    print("content_id:",content_id[:2])
    print("keyword_id:",keyword_id[:2])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    title_pad = kr.preprocessing.sequence.pad_sequences(title_id, title_max_length)
    content_pad = kr.preprocessing.sequence.pad_sequences(content_id, content_max_length)
    keyword_pad = kr.preprocessing.sequence.pad_sequences(keyword_id, keyword_max_length)
    if not test:
        y_pad = kr.utils.to_categorical(df['label'].tolist(), num_classes=2)  # 将标签转换为one-hot表示

    print("title shape:",title_pad.shape)
    print("content shape:",content_pad.shape)
    print("keyword shape:",keyword_pad.shape)
    print("auxilary shape:",auxilary.shape)
    # print("y shape:",y_pad.shape)
    if not test:
        return title_pad, content_pad, keyword_pad, auxilary, y_pad
    else:
        return title_pad,content_pad,keyword_pad,auxilary