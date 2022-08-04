#!/usr/bin/env python
# coding: utf-8

from match import *

if __name__ == "__main__":
    data_path = '/content/drive/MyDrive/datayes.xlsx'
    data = load_data(data_path)

## 导入预训练gpt2模型和以此计算的全部标题向量
    model_0 = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/gpt2small_chinese")
    TitleEmbeddingTensor_0 = torch.load('/content/drive/MyDrive/TitleEmbeddingTensor_base.pt')
## 导入微调后的gpt2和以此计算的全部标题向量
    device = torch.device("cpu")
    model_1 = torch.load('/content/drive/MyDrive/minGPT2/model_gpt2.pt',map_location=device)
    TitleEmbeddingTensor_1 = torch.load('/content/drive/MyDrive/TitleEmbeddingTensor_finetune.pt')
    tokenizer_gpt2 = BertTokenizer.from_pretrained("/content/drive/MyDrive/gpt2small_chinese") 
## 导入预训练好的simcse模型
    encoder = load_model('/content/drive/MyDrive/encoder_simcse.h5',custom_objects={'simcse_loss': simcse_loss})
# 预先将全部的标题集转化为token，并进行预测和标准化
    t_simcse = np.load("/content/drive/MyDrive/title_token_simcse.npy",allow_pickle = True)
    dict_path = '/content/drive/MyDrive/chinese_simbert_L-12_H-768_A-12/vocab.txt'
    tokenizer_simcse = Tokenizer(dict_path, do_lower_case=True, pre_tokenize=None) 

    Match(data.iloc[109],usage='simcse',index1=109,threshold=0.77)
    T1vsT2(data.iloc[109],data.iloc[106],usage='simcse')

