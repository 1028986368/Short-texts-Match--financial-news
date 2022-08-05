# 短文本匹配——财经新闻
## 使用说明
### 1、判断一条新闻是否已经在已有的新闻标题库中
默认为新闻标题库为1月~7月的财经新闻，若想要更换，需要重新训练
```
python main.py
# usage='simcse' or 'gpt2_base' or 'gpt2_finetune' ;
# threshold=[0,1]
Match(title,'simcse',0.77)
```

### 2、判断两条新闻是否相同
```
python main.py
T1vsT2(title1,title2,usage) #usage='simcse'or'gpt2_base'or'gpt2_fintune'
```

## 训练说明
```
# 修改预训练模型路径，数据路径
python train.py
```

### 预训练模型
1、gpt2small_chinese  
https://drive.google.com/drive/folders/1eerX1N8n_eFlnQ4xpxZ4iU2-Mx83pXFp?usp=sharing  
2、chinese-simbert模型  
https://drive.google.com/file/d/1phCkA5tsTm6IFX7BKq6NGcQud6o_S-yd/view?usp=sharing

### 微调后的模型
3、经过微调的gpt2模型  
https://drive.google.com/file/d/1phCkA5tsTm6IFX7BKq6NGcQud6o_S-yd/view?usp=sharing  
4、经过微调的simcse模型  
https://drive.google.com/file/d/1phCkA5tsTm6IFX7BKq6NGcQud6o_S-yd/view?usp=sharing

### 数据说明
1、新闻标题库  
https://docs.google.com/spreadsheets/d/1J0Z-4ZwpscPa6AftCgvmqeWl-2Y8DAHz/edit?usp=sharing&ouid=104926295998652945461&rtpof=true&sd=true  
2、根据微调后的simcse，计算的title_token  
https://drive.google.com/file/d/1e06SSYIQBH2_4hFiC6q7rFo81Whtas_C/view?usp=sharing  
3、根据微调后的gpt2，计算的title_token  
https://drive.google.com/file/d/1vPPQUyvBC-6Qp_t60e2zh9c9aMs_iRSp/view?usp=sharing  
4、根据预训练模型gpt2，计算的title_token  
https://drive.google.com/file/d/1E8JLG42NhiiVKFfh8ztNYrh_vCKNb_pd/view?usp=sharing  
 
 
注：2-4提前计算好，便于后面判断一条新闻是否已经在已有的新闻标题库中时，快速、更稳定
  

#### 环境
```
h5py==2.10.0
simpletransformers==0.32.3
tensorflow-gpu==2.0
transformers==4.21.0
bert4keras==0.11.3
torch==1.12.0
```
