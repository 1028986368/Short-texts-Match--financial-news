# -*- coding: utf-8 -*-
"""
author: jolly_zhang
all about function
"""

import jieba
from tqdm import tqdm
from gensim import corpora, similarities,models
from pprint import pprint
import re
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import metrics
import matplotlib.pyplot as plt
from transformers import (GPT2Config,GPT2LMHeadModel,GPT2Tokenizer)
from transformers import BertTokenizer
import torch
from string import punctuation as pnc
from collections import Counter
from scipy import spatial
from bs4 import BeautifulSoup
import pylab as pl
import datetime as dt
pd.set_option('display.max_colwidth', -1)
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.backend import K,keras
from keras.models import load_model
import tensorflow as tf
from bert4keras.layers import *

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

def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

# 绘制ROC曲线，便于判断阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
def ROC(label, y_prob):
    s = len(label)*[1]
    for i in range(len(label)):
        if label[i]==0:
            s[i]=19
    s = np.array(s)
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob,sample_weight=s)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point
# 构建gpt2模型
def bulid_model_gpt2(detail):
    if detail == 0:
        model = model_0
        TitleEmbeddingTensor = TitleEmbeddingTensor_0

    elif detail == 1:
       model = model_1
       TitleEmbeddingTensor = TitleEmbeddingTensor_1
    embeddigs = model.transformer.wte
    tokenizer = tokenizer_gpt2
    return TitleEmbeddingTensor,embeddigs,tokenizer

# gpt2计算标题相似度
def getMostSimilarQuestionsIdx(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1)).squeeze(0)
    res = res.tolist()
    return res

def simcse_loss(y_true, y_pred):
    """
    用于SimCSE训练的loss
    """
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

# 构建simcse模型
def bulid_model_simcse():
    tokenizer =  tokenizer_simcse
    return tokenizer

## 转化为词向量
def doc2token(data):
    tokenizer = bulid_model_simcse()
    token_ids = tokenizer.encode(data, maxlen=64)[0]
    return token_ids

## 计算相似度
def sim_cos_simcse(a_vecs,b_vecs):
    a = doc2token(a_vecs)
    a = np.array([a])
    a_vecs = encoder.predict([a,np.zeros_like(a)],verbose=False)
    a_vecs = l2_normalize(a_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    return sims

# 修正标题相似度
def fix(title,title_dict):
    if title['author'] is not None and title['media'] is not None:
        for i in range(10):
            if title_dict[i][1]>0.999:
                if title['author']==data['author'][title_dict[i][0]] and title['media']==data['media'][title_dict[i][0]] and title['url']!=data['url'][title_dict[i][0]]:
                     title_dict[i] = (title_dict[i][0],0)

# 计算标题与已有标题库近两天标题的相似度
# title（字典型，包含的key与data保持一致）
def similar_TF(title,usage='simcse',index1=None):
    a = dt.timedelta(days=2)
    b = title['create_time'].date()
    l = re.findall('感染者+|病例+',title['title'])
  
    if len(l)>0:
        data_belong = data[(data['create_time']>=dt.datetime(b.year,b.month,b.day,0,0,0))&(data['create_time']<=title['create_time'])]
    else:
        data_belong = data[(data['create_time']>=(title['create_time']-a)) & (data['create_time']<=title['create_time'])]

    if index1 is not None:
        data_belong = data_belong.drop(index=index1)



    elif usage == 'gpt2_base':
        TitleEmbeddingTensor,embeddigs,tokenizer = bulid_model_gpt2(0)
        input = title
        preprocessedinput = input['title']
        inputEncoded = tokenizer.batch_encode_plus([preprocessedinput])['input_ids']
        inputEmbedded1 = embeddigs(torch.tensor(inputEncoded).to(torch.int64)).squeeze(0).mean(axis=0).unsqueeze(0)
        t = [TitleEmbeddingTensor[i] for i in data_belong.index]
        numQ = len(t)
        t = torch.cat(t,dim=0)
        embedDim = 768
        t = torch.reshape(t, (numQ, embedDim))
        title_dict = getMostSimilarQuestionsIdx(inputEmbedded1,t)
        title_dict = dict(zip(data_belong.index,title_dict))
  
    elif usage == 'gpt2_finetune':

        TitleEmbeddingTensor,embeddigs,tokenizer = bulid_model_gpt2(1)
        input = title
        preprocessedinput = input['title']
        inputEncoded = tokenizer.batch_encode_plus([preprocessedinput])['input_ids']
        inputEmbedded = embeddigs(torch.tensor(inputEncoded).to(torch.int64)).squeeze(0).mean(axis=0).unsqueeze(0)
        t1 = [TitleEmbeddingTensor[i] for i in data_belong.index]
        numQ = len(t1)
        t1 = torch.cat(t1,dim=0)
        embedDim = 768
        t1 = torch.reshape(t1, (numQ, embedDim))
        title_dict = getMostSimilarQuestionsIdx(inputEmbedded,t1)
        title_dict = dict(zip(data_belong.index,title_dict))


    elif usage == 'simcse':
        kk = np.array([t_simcse[i] for i in data_belong.index])
        title_dict = sim_cos_simcse(title['title'],kk) #{}
        title_dict = dict(zip(data_belong.index,title_dict))

        title_dict = list(title_dict.items())
        title_dict.sort(key=lambda x:x[1],reverse=True)

    if title_dict[0][1]>0.99:
        fix(title,title_dict)
  
    title_dict.sort(key=lambda x:x[1],reverse=True)

    return title_dict

# 判断标题是否已在标题库中
def Match(title,usage='simcse',index1=None,threshold=0.77):
    title_dict = similar_TF(title,usage,index1)
    i = 0
    l = []
    while title_dict[i][1] > threshold:
        l.append([title_dict[i][0],data['title'][title_dict[i][0]],title_dict[i][1]])
        i=i+1
    print(tabulate(l, headers=["id", "title", "similarity"]))