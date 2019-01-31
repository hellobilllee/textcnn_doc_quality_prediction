# coding: utf-8
from __future__ import print_function
import datetime
import os
from data_helper import read_vocab, word_cutter, get_content_from_htmlpage, process_file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from cnn_model import TCNNConfig, TextCNN
import os

vocab_dir = "data/w2v/qttnews.vocab.txt"
word_vector_dir = "data/w2v/qttnews.vector.txt"
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
tf.reset_default_graph()
class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.config.pre_trianing = pd.read_csv(word_vector_dir, header=None, index_col=None).values
        self.model = TextCNN(self.config)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.session = tf.Session(config=session_conf)
        self.session.run(tf.global_variables_initializer())

        # self.session.run(tf.initialize_local_variables())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def parse_json_to_df(self,json_message):
        samples = []
        feature_names = ['create_time', 'content_id', 'detail', 'title', 'keywords', 'type', 'rank']
        for sample in json_message["data"]:
            samp = []
            for feat in feature_names:
                samp.append(sample.get(feat))
            samples.append(samp)
        df_samples = pd.DataFrame(data=samples, columns=feature_names)
        return df_samples

    """
    特征构造： 
    create_time,current_time: publish_to_update_hour
    type: one-hot
    rank: one-hot
    title: title_length,title_token_length,vectorize 
    detail: content_length,content_token_length,vectorize 
    keywords: keyword_length,vectorize 
    """
    def predict(self, json_message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        df = self.parse_json_to_df(json_message)
        type_cat = [16, 1, 4, 42, 9, 7, 8, 6, 17, 19, 25, 5, 12, 13, 23, 10, 27, 15, 14, 18]
        rank_cat = [1, 2, 3, 4, 5]
        df_notype = df[~df['type'].isin(type_cat)][["content_id"]]
        df_norank = df[~df['rank'].isin(rank_cat)][["content_id"]]
        if len(df_notype):
            df_notype["score"] = 0
        if len(df_norank ):
            df_norank["score"] = 0
        df = df[df['type'].isin(type_cat)]
        df = df[df['rank'].isin(rank_cat)]


        #如果数据当中含有nan值，则取出这些content_id
        df_null_content_id = df[df.isnull().any(axis=1)][["content_id"]]
        if len(df_null_content_id):
            df_null_content_id["score"] = 0
        df = df[~df.isnull().any(axis=1)]

        #去除detail中的html标签
        df["detail"] = df['detail'].apply(get_content_from_htmlpage)
        # print(df['detail'])
        df['create_time'] = pd.to_datetime(df['create_time'])
        current_time = datetime.datetime.now()
        df["update_time"] =current_time
        df_entertain_sports = df[df.type.isin([6,13])][["content_id","create_time"]]
        if len(df_entertain_sports):
            df_entertain_sports["create_time"] = df_entertain_sports["create_time"].apply(lambda x: current_time-x)/np.timedelta64(1,'h')
            df_entertain_sports["create_time"] = df_entertain_sports["create_time"].apply(lambda x: 1.0 if x<=4 else 1.0/math.log(x,4) )
            df_entertain_sports_dict = dict(zip(df_entertain_sports["content_id"].tolist(), df_entertain_sports["create_time"].tolist()))
        # df['weekday'] = df['create_time'].dt.dayofweek
        # cal = calendar()
        # # print(df['create_time'])
        # holidays = cal.holidays(start=df['create_time'].min(), end=df['create_time'].max())
        # df['holiday'] = df['create_time'].isin(holidays).apply(lambda x: 1 if x else 0)
        df["publish_to_update_hour"] = (df['update_time'] - df['create_time']) / np.timedelta64(1, 'h')
        df_too_old = df[df["publish_to_update_hour"] > 36][["content_id"]]
        if len(df_too_old):
            df_too_old["score"] = 0
        df = df[df["publish_to_update_hour"] <= 36]
        df["publish_to_update_hour"] = df["publish_to_update_hour"].apply(lambda x: 1.0 if x <= 3 else math.log(x, 3))

        df.drop(labels=['create_time',"update_time"], axis=1, inplace=True)

        wc = word_cutter("data/dict")
        df['title_length'] = df['title'].apply(lambda x: len(wc.clean_html(x)))
        df['title'] = df['title'].apply(lambda x: wc.cut_words(x))
        df["title_token_length"] = df['title'].apply(lambda x: len(x.split(" ")))
        df['detail_length'] = df['detail'].apply(lambda x: len(wc.clean_html(x)))
        df['detail'] = df['detail'].apply(lambda x: wc.cut_words(x))
        df['detail_token_length'] = df['detail'].apply(lambda x: len(x.split(" ")))
        # print("after tokenization\n",df.describe(include="all"))
        df_short_length_content_id = df[((df['title_length'] <= 1) | (df['title_token_length'] <=1)) | ((df['detail_length'] <= 10)| (df['detail_token_length'] <=3))][["content_id"]]

        if len(df_short_length_content_id):
            df_short_length_content_id["score"] = 0
        df = df[((df['title_length'] > 1) & (df['title_token_length'] > 1)) & ((df['detail_length'] > 10)& (df['detail_token_length'] > 3))]
        df['detail_length'] = df['detail_length'].apply(lambda x: math.log2(x))

        print(df.info())
        print("begin to make predictions")
        title_test, content_test, keyword_test, auxilary_test = process_file(df, self.word_to_id,
                                                                                             self.config.title_seq_length,
                                                                                             self.config.content_seq_length,
                                                                                             self.config.keyword_seq_length,test=True)

        feed_dict = {
            self.model.input_x_title: title_test,
            self.model.input_x_content: content_test,
            self.model.input_x_keyword: keyword_test,
            self.model.input_x_auxilary: auxilary_test,
            self.model.keep_prob: 1.0
        }

        y_pred_prob = self.session.run(self.model.y_pred_prob, feed_dict=feed_dict)
        df["score"] = pd.Series(y_pred_prob)
        df_result = df[["content_id","score"]]
        if len(df_notype):
            df_result = pd.concat([df_result,df_notype],axis=0)
        if len(df_norank):
            df_result = pd.concat([df_result,df_norank],axis=0)
        if len(df_null_content_id):
            df_result = pd.concat([df_result,df_null_content_id],axis=0)
        if len(df_too_old):
            df_result = pd.concat([df_result,df_too_old],axis=0)
        if len(df_short_length_content_id):
            df_result = pd.concat([df_result,df_short_length_content_id],axis=0)
        if len(df_entertain_sports):
            for index,row in df_result.iterrows():
                if row["content_id"] in df_entertain_sports_dict.keys():
                    df_result.loc[index, "score"] = df_entertain_sports_dict.get(row['content_id'])* df_result.loc[index, "score"]
        df_result["score"] = df_result["score"]*1000
        print(df_result.describe())
        tf.reset_default_graph()
        return df_result.to_json(orient='records')

def run_predict(json_message):
    cnn_model = CnnModel()
    return cnn_model.predict(json_message)

if __name__ == '__main__':

    # cnn_model = CnnModel()
    test_demo = {'data': [{'content_id': 152787347,
   'create_time': '2019-01-20 10:07:59',
   'detail': '<p>吉林的长白山，最早在4000多年前的《山海经》里就记录了它，这里是我国东北地区的最高峰，也鸭绿江、松花江、图们江的发源地，也是中朝两国的界山。这次长白山之行，由于大雪的原因，很遗憾没有看到天池，于是便改道去了长白山大峡谷。长白山大峡谷，现在叫做锦江大峡谷，也是近些年才发现的一处世界级的大峡谷，在90年代的时候，长白山大峡谷就已被发现，只是近几年被开发成了游客游览的目的地。火山爆发后熔岩石表面的火山灰和泥土被江水及雨水冲刷所致。北锦江从谷底流下，两岸怪石林立，奇景叠生，是国内规模最大的火山岩区峡谷地貌。长白山是一座休眠火山，在大约2500万年的时间里，长白山地区经历了几次火山喷发活动，不过现在属于稳定时期，也正是因为它是一座休眠火山，所以在海拔两千多米的山上，有多处温泉不断从地下涌出。林海里到处都是近一尺深的积雪，我们短暂的兴奋之后就会觉得非常冷，尤其是在有风的时候，哪怕你穿再多的衣服都不会觉得很暖和。幸好这会还有阳光。长白山的“长白”二字还有一个很美好的寓意，即“长相守，到白头”。如果是冬天到长白山，便真的可以体验到“长相守，到白头”，因为在大雪纷飞里，每个人的头顶不一会儿就会变得白花花。处于北纬42度的长白山，雪属于粉雪，粉雪并不是粉色的雪，而是指没有冰冻就落到了地面的雪绒花，所以特别松软有弹性，挤压后能像海绵一样恢复原状，所以玩雪的时候抛起来也特别松软。这场雪可把这位南方来的小妹妹给乐坏了。玩的不亦乐乎，把那双小手套都冻的通红。长白山锦江大峡谷全长达70公里，宽约200-300米，深约80-100米，两岸坡陡如削，但在冬日大雪的覆盖下，银装素裹，就好似一幅安静的水墨画卷。照片中有块地方像不像两条鱼呀？有嘴有眼的。大峡谷是由火山堆积经1000多年的断裂构造作用和流水侵蚀切割而形成的，两岸生长着茂密的大森林，树木笔直粗壮，在树林里徒步，经常能发现蘑菇和小动物。此刻的夕阳透过身边的树林在雪地上投射出长长的影子。雪地上的警示牌，这方面景区做的很不错，景色再美，安全也永远是第一位的。通常凡是峡谷，谷底都会有河水流过，就像美国的科罗拉多大峡谷，锦江大峡谷也不例外。别看气温是零下，峡谷中的小河流并没有冻住，汩汩而下，可见这里是有地热的。因为雪厚，不知道下面有多深，很可能就会陷进去。游客只能在林间的木栈道上行走，在栈道两侧是禁止下去的，偶而有小伙伴试着踩了几下雪，然后拍张照片，便又马上回来了。一个美女在拍照，就会吸引旁边一堆长枪短炮的扫射。拍美女也是很多摄客们的一大爱好。看到了要是不拍几张就跟吃了多大亏似的。之前在夏天的时候也来过这里，很凉快的一个地方，在这里每个季节都会呈现出它不一样的美。在现在的大雪中，这里就成了一片银装素裹。<p>',
   'keywords': '长白山,大峡谷,锦江大峡谷,休眠火山,粉雪,大雪,两岸,峡谷',
   'rank': 3,
   'title': '奇景叠生，怪石林立，看看雪中的长白山大峡谷有多美 ',
   'type': 16,
   'validread': 98,
   'validshow': 1006},
  {'content_id': 152970931,
   'create_time': '2019-01-20 10:07:59',
   'detail': '<p>中国外交部：2018年共处理领事保护和协助案件约8万起 中新社北京1月9日电 (黄钰钦 蒋涛)中国外交部领事司司长郭少春9日表示，2018年外交部和驻外使领馆会同各有关部门妥善处理领事保护和协助案件约8万起，案件数量较上年增长1万起。    外交部新闻发言人陆慷  当日，中国外交部2018年度领事工作媒体吹风会在北京举行，郭少春在介绍维护海外中国公民和企业安全与合法权益工作时作上述表示。 他指出，2018年外交部领事司日常协助类案件有所减少，反映出中国公民在海外出行方面的安全意识不断增强，对国际旅行常识的理解也越来越充分。“值得注意的是，涉及人数多、社会关注度高的大案要案有所增加。包括泰国普吉岛游船倾覆、印尼龙目岛地震、美国塞班岛遭台风袭击致中国游客滞留等重大案件均具有涉及面广、事发突然、处置难度较大等特点。”郭少春说。 针对如何应对海外中国公民违法犯罪问题，郭少春强调要注重把握领事保护与协助工作的“可为”与“不可为”，坚持“合法权益受保护，违法行为不袒护”的工作原则。 他表示，对极少数人在海外参与甚至组织网络赌博、实施跨境电信诈骗、走私贩运濒危野生动植物及其制品等严重违法犯罪活动，中国坚决支持有关国家依法严肃处理，对涉案中国籍人员仅限于提供最低限度领事协助。“领事保护必须尊重驻在国的法律法规，坚决不能为极个别的违法犯罪行为买单。<p>”',
   'keywords': ' 领事,郭少,协助,保护,中国公民,外交部,案件,中国外交部',
   'rank': 1,
   'title': '外交部：2018年共处理领事保护和协助案件约8万起 ',
   'type': 1,
   'validread': 98,
   'validshow': 1006},
  {'content_id': 152773652,
   'create_time': '2019-01-20 10:07:59',
   'detail': '<p>2018物业服务收费包括哪些内容 ,入住小区就要遵从物业管理办法，2018物业服务收费包括哪些内容？接下来，且听PChouse细细说来。1、公共物业及配套设施的维护保养费用。2、聘用管理人员的薪金。3、公用水电的支出。4、购买或租赁必需的机械及器材的支出。5、物业财产保险（火险、灾害险等）及各种责任保险的支出。6、垃圾清理、水池清洗及消毒灭虫的费用。7、清洁公共地方及幕墙、墙面的费用。8、公共区域植花、种草及其养护费用。9、更新储备金。10、聘请律师、会计师等专业人士的费用。11、节日装饰的费用。12、管理者酬金。13、行政办公支出。14、公共电视接收系统及维护费用<p>',
   'keywords': '支出,费用,物业,服务收费,公共,电视接收,植花,聘请律师',
   'rank': 5,
   'title': '2018物业服务收费包括哪些内容',
   'type': 8,
   'validread': 98,
   'validshow': 1006},
  {'title': '《皓镧传》聂远化身战国“成功学讲师”',
   'validshow': 0,
   'detail': '<p><img data-src="http://static.1sapp.com/lw/img/2019/01/31/e3d8c62c6404bddb134b748eedf04a91.jpeg" data-size="520,320"></p><p>由吴谨言，聂远主演的《皓镧传》近期正在热播，播出两周后，各领域讨论度居高不下，明晚《皓镧传 》继续迎来更新!吕不韦深陷秦国，为秦王孙谋划大业，而在赵国，吕不韦遇刺身亡的消息却让李皓镧崩溃不已。更多精彩剧情即将揭晓!</p><p>《皓镧传》剧情播出还没过半，讨论却在各个圈层进行的如火如荼，跌宕起伏的故事情节，纷杂繁复的人物关系引人入胜，同时，剧中所表达的家国情怀，也引起了诸多关注。预告中可以看到，本周即将更新的剧情中，吕不韦以身试险，亲赴秦 国，为秦王孙嬴异人谋划大业，但在前往秦国途中却遭到公子蛟的暗算，吕不韦机智脱险后即刻奔赴秦国，但在赵国，传回来的却是吕不韦已经身亡的消息，李皓镧万念俱灰，几近崩溃……不离cp 的发展时刻牵动着剧粉们的心，看到预告后，众多剧粉都惊呼：不离cp请不要be!</p><p>除了剧情十分引人关注之外，剧中的不少台词也是深入人心，聂远饰演的吕不韦，在剧中时刻提点督促李 皓镧，也因为其“话痨”的人设被网友们封为“战国成功学讲师”，剧中略显鸡汤的台词，着实让大家看到了当时毒舌傲娇大猪蹄子的影子，但同时也被各路的KOL争相转发，并配以：转发这个吕boy 你将会获得成功等文字!一时间，聂远饰演的吕不韦，成为了剧粉们的新宠!</p>',
   'content_id': 158470470,
   'keywords': '吕不韦,皓镧传,聂远,李皓,剧粉,秦国,剧情,成功学',
   'validread': 0,
   'create_time': '2019-01-31 13:06:36',
   'rank': 1,
   'type': 6},
  {'title': '福建晋江警方破获“1040阳光工程”传销案，抓获266人',
   'validshow': 0,
   'detail': '<p>近日，福建晋江 市公安局破获了一起利用资金盘运作的组织、领导传销案件，抓获各层级传销人员266名，并查获大量用于传销活动的宣传资料、书籍等，涉案金额达一千多万元。2019年1月30日，晋江市公安局 召开新闻发布会，对外通报了这起传销案的相关案情。</p><p><img data-src="http://static.1sapp.com/lw/img/2019/01/31/bb1db76c2160d96b1cadb7f76e9ba34b.jpeg" data-size="1024,1024"></p><p>微博@晋江公安 图</p><p>鼓吹投入数千元最终回报1040万元</p><p>2018年10月20日，晋江市公安局接福建省公安厅转来的线索称，一名吴姓男子报警称其嫂子张某可能陷入传销，地址在晋江池店。</p><p>2018年10月21日凌晨，晋江警方在池店镇桥南片区的一个小区租房内找到张某，一并被找到的还有张某的父亲张某智。</p><p>今年52岁的张某智是湖北人，原本到晋江是帮 侄子照看孩子，2018年6月，他经人介绍，交了4000元投资了一个资金盘。“交完钱之后，就有人带我去听课、聊天，一般都是三、四个人跟我讲课，每个月差不多是听2次课。”张某智说，他并不 知道资金盘具体怎么运作，只知道要拉人头投资。</p><p>女儿张某就是张某智拉进来的，张某今年29岁，湖北人，育有一儿一女，最大的儿子不过6岁，在上幼儿园。因婆媳产生矛盾，张某本就 想着离开老家，刚好父亲在晋江让她一起来投资资金盘，她便来到了晋江，也投了4000元。</p><p>张某来到晋江后，还想拉上丈夫一起来，想向公公借10万元。但张某丈夫不肯，怀疑妻子陷入传销，将此事告知了其弟弟、报警人吴某。吴某向公安机关报了警。</p><p>张氏父女投资所谓资金盘，只需要他们去听课，拉人头、缴纳会费、发展下线，是典型的传销活动特征。泉州市公安局党委委员、晋江市公安局局长陈文荣立即指令苏琪副局长牵头组织，成立由指挥中心、经侦大队、网安大队、池店派出所、紫帽派出所等单位组成的工作专班，立案展开调查。</p><p>经查，一个以徐某龙为首的所谓“1040阳光工程”传销组织浮出水面，该传销组织分为五级三晋制，按投资份额分业务员、组长、主任、经理、老总五个级别，按照投资的份额和下线的人数、级别分三个晋升阶 段：晋升主任、晋升经理、晋升老总。</p><p>传销组织要求参加者最低缴纳3800元，最高缴纳69800元获得加入资格，鼓吹最终回报1040万元欺骗成员，鼓动成员积极发展下线购买份额以牟取非 法利益，并按照一定顺序组成层级，直接或间接以发展人员数量作为计酬的依据，引诱参加者继续发展他人参加，骗取财物，扰乱经济社会秩序。</p><p>出动800多警力抓获266人</p><p>2018年11月26日晚，晋江市公安局组织27个局属单位、派出所警力800多名，在晋江池店、紫帽辖区开展收网行动，抓获各层级传销人员266名，并查获大量用于传销活动的宣传资料、书籍等，涉案金额高达1000多万元。</p><p>据办案民警介绍，在抓获的犯罪嫌疑人里，有一名“老总”张某涛（男，37岁，湖北人），一名体系自律大总管徐某龙（男，33岁，安徽人），还有不少是“家长”。</p><p> 办案民警说，传销组织内部以租住房为单位，设立了诸多小团体家庭，由“家长”负责照看其他成员并组织互相学习。民警捣毁的传销活动窝点就是租住房。</p><p>“传销组织基本都是靠‘杀熟’”，办案民警说，他们打着 “自愿连锁经营”的旗号，采取编造、歪曲国家政策，虚构、夸大经营、投资、服务项目及盈利前景等方式，从身边熟人下手，诱骗他人来到晋江，并交纳3800到69800元不 等的费用加入传销组织。</p><p>据民警介绍，这个传销组织套路是先把亲戚或朋友骗来，带着这些亲戚或朋友游山玩水热情招待两、三天，以此拉近关系，之后带着亲戚或朋友去听课，向他们灌输快速发财致富的思想，骗他们缴纳会费，交了会费的会员会被归入相应的“家庭”，由“家长”统一管理，传授他们发展下线的方法，拉更多的人下水。</p><p>目前，有36人被警方刑事拘留，2人 被监视居住；晋江检察院对情节较重的18人做出批准逮捕决定，其他人员做出直诉等处理。</p><p>晋江警方提醒，无论传销的“外衣”怎么换，要辨别传销的真面目，只需看三个特征：第一，是否需要认购商品或交纳费用取得加入资格；第二，是否需要发展他人成为自己的下线，形成层级关系；第三，是否以直接或间接发展人员的数量为依据计算报酬（奖金）。只要符合这三个特征，就 肯定是传销，千万要提高警惕，防止上当受骗。</p>',
   'content_id': 158469582,
   'keywords': '晋江,张某,传销,传销组织,池店,晋江市,传销案,下线',
   'validread': 0,
   'create_time': '2019-01-31 13:00:38',
   'rank': 1,
   'type': 1}]}

    # print(run_predict(test_demo))
    # print(cnn_model.predict(test_demo))
