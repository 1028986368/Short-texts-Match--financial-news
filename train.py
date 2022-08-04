# -*- coding: utf-8 -*-
"""train.ipynb

"""

from numpy.random import choice
import random
import datetime as dt
from tqdm import tqdm
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding
from keras.models import Model
import sys
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
jieba.initialize()
random.seed(12345)

def load_data(data_path):
    
  # 加载数据  
    df = pd.read_excel(data_path)
    data = df.values
    data = pd.DataFrame(data)
    data.columns = ['id','newsid','title','url','media','source','s3','keywords','create_time','pubdate','update_time','author','mediatype','stock','summary','topicid','topic_name','topicurl','docstatus','pushstatus','columnid','COLUMNNAME','PICURL','DOCID','SUBTITLE','FEEDPIC','RECEIVETIME','TASKSTATUS','ETL_CRC','QA_RULE_CHK_FLG','QA_MANUAL_FLG','QA_ACTIVE_FLG','CREATE_BY','UPDATE_BY','TMSTAMP']
    data = data.drop(['topicid','topic_name','topicurl','docstatus','pushstatus','columnid','COLUMNNAME','PICURL','DOCID','SUBTITLE','FEEDPIC','RECEIVETIME','TASKSTATUS','ETL_CRC','QA_RULE_CHK_FLG','QA_MANUAL_FLG','QA_ACTIVE_FLG','CREATE_BY','UPDATE_BY'],axis=1)
  ## 删除乱码的新闻标题
    data = data.drop([11836, 11838, 11839, 11843, 11844, 11845],axis=0)
    data.index = range(99994)
  # 转换日期格式
    data['create_time_c'] = data.create_time.values.astype("datetime64[D]")
  # 将NAN 转化为None
    ll = data
    data = data.where(data.notnull(), None) 
    return data

# 随机抽取100条新闻，与其前2天新闻组成训练数据集

def trans(title):
    a = dt.timedelta(days=2)
    data_belong = data[(data['create_time']<title['create_time']) & (data['create_time']>(title['create_time']-a))]   
    return data_belong

def samples_train(data):
    ll=[]
    samples = choice(range(99994), size=100,replace=False)
    data_sample = data[['title']].iloc[samples]
    for i in data_sample.index:
        data_belong = trans(data.iloc[i])
        for j in data_belong.index:
            ll.append((data_sample['title'][i],data['title'][j],0)) 
    random.shuffle(ll)

# 建立分词器
def get_tokenizer(dict_path, pre_tokenize=None):
    return Tokenizer(dict_path, do_lower_case=True, pre_tokenize=pre_tokenize)

# 建立编码器
def get_encoder(config_path,checkpoint_path,model='bert',pooling='first-last-avg',dropout_rate=0.1):

    bert = build_transformer_model(
            config_path,
            checkpoint_path,
            model=model,
            with_pool='linear',
            dropout_rate=dropout_rate)

    outputs, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            outputs.append(output)
            count += 1
        except:
            break
    output = keras.layers.Lambda(lambda x: x[:, 0])(outputs[-1])
    encoder = Model(bert.inputs, output)
    return encoder

# 转换文本数据为id形式
def convert_to_ids(data, tokenizer, maxlen=64):
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids, b_token_ids, labels

# 标准化
def l2_normalize(vecs):
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

# Spearman相关系数
def compute_corrcoef(x, y):
    return scipy.stats.spearmanr(x, y).correlation
# 用于SimCSE训练的loss
def simcse_loss(y_true, y_pred):
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)

class data_generator(DataGenerator):
    """
    训练语料生成器
    """
    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []

if __name__ == "__main__":
  # 构建训练数据集
    data_path = '/content/drive/MyDrive/datayes.xlsx'
    data = load_data(data_path)
    ll = samples_train(data)
    dataset = {'STS-B-train':ll[:100000],'STS-B-test':ll[100000:110000],'STS-B-valid':ll[110000:]}
  # 基本参数
    model_type, pooling, task_name, dropout_rate = 'SimBERT','cls','STS-B','0.3'
    dropout_rate = float(dropout_rate)
    maxlen = 64
    datasets = dataset
    model_name = {
    'SimBERT': 'chinese_simbert_L-12_H-768_A-12',
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'}[model_type]
    config_path = '/content/drive/MyDrive/%s/bert_config.json' % model_name
    checkpoint_path = '/content/drive/MyDrive/%s/bert_model.ckpt' % model_name
    dict_path = '/content/drive/MyDrive/%s/vocab.txt' % model_name
  # 建立分词器
    tokenizer = get_tokenizer(dict_path)
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate）
  # 语料id化
    all_names, all_weights, all_token_ids, all_labels = [], [], [], []
    train_token_ids = []
    for name, data in datasets.items():
        a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
        all_names.append(name)
        all_weights.append(len(data))
        all_token_ids.append((a_token_ids, b_token_ids))
        all_labels.append(labels)
        train_token_ids.extend(a_token_ids)
        train_token_ids.extend(b_token_ids)

    if task_name != 'PAWSX':
        np.random.shuffle(train_token_ids)
        train_token_ids = train_token_ids[:10000]
  
  # SimCSE训练
    encoder.summary()
    encoder.compile(loss=simcse_loss, optimizer=Adam(2e-5))
    train_generator = data_generator(train_token_ids, 64)
    encoder.fit(
    train_generator.forfit(),steps_per_epoch=len(train_generator), epochs=1)

    # save_model(encoder,'/content/drive/MyDrive/encoder_simcse.h5')  # 将训练好的模型保存为HDF5文件
