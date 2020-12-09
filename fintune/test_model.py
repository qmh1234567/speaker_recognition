
from glob import glob
import os
import numpy as np
import pandas as pd
import keras.backend as K

from random_batch import data_catalog
from eval_metrics import evaluate
from triplet_loss import deep_speaker_loss
import tensorflow as tf
import pickle

import sys
sys.path.append("../") 
import usedModels.Att_DCNN as Att_DCNN

# # OPTIONAL: control usage of GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.compat.v1.Session(config=config)


BATCH_SIZE = 32
TRIPLET_PER_BATCH = 3  # 选3个人
TEST_DIR = "/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/test" # 测试集目录
TEST_NEGATIVE_No = 99  # 负语音样本的人数  99好像跑不下来

num_neg = TEST_NEGATIVE_No

# 构建测试数据集
def create_test_data(dataset,check_partial):
    global num_neg
    # dataset = data_catalog(test_dir)
    unique_speakers = list(dataset['speaker_id'].unique())
    np.random.shuffle(unique_speakers)
    num_triplets = len(unique_speakers)
    if check_partial:
        num_neg= TEST_NEGATIVE_No-int((TEST_NEGATIVE_No+1)/2)
        num_triplets = min(num_triplets,30)
    test_batch = None
    for i in range(num_triplets):
         # 构建anchor
        anchor_positive_file = dataset[dataset['speaker_id'] == unique_speakers[i]]
        if len(anchor_positive_file) <2:
            continue
        anchor_positive_file = anchor_positive_file.sample(n=2, replace=False)
        anchor_df = pd.DataFrame(anchor_positive_file[0:1])
        anchor_df['training_type'] = 'anchor'   #   1 anchor，1 positive，num_neg negative
        
        if test_batch is None:
            test_batch = anchor_df.copy()
        else:
            test_batch = pd.concat([test_batch, anchor_df], axis=0)
            
        # 构建positive
        positive_df = pd.DataFrame(anchor_positive_file[1:2])
        positive_df['training_type'] = 'positive'
        test_batch = pd.concat([test_batch, positive_df], axis=0)
        
        # 构建negative
        negative_files = dataset[dataset['speaker_id'] != unique_speakers[i]].sample(n=num_neg, replace=False)
        for index in range(len(negative_files)):
            negative_df = pd.DataFrame(negative_files[index:index+1])
            negative_df['training_type'] = 'negative'
            test_batch = pd.concat([test_batch, negative_df], axis=0)
            
    # test_batch.to_csv("test_dataset.csv")
    return to_inputs(test_batch,num_triplets)

# 构建输入和输出
def to_inputs(dataset_batch,num_triplets):
        new_x = []

        for i in range(len(dataset_batch)):
            filename = dataset_batch[i:i + 1]['filename'].values[0]
            with open(filename,"rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                new_x.append(x)

        x = np.array(new_x) #（1530，299，40，1）
        # y = dataset_batch['speaker_id'].values  #（1530）
        new_y = np.hstack(([1,1],np.zeros(num_neg)))  # 1 positive, num_neg negative
        y = np.tile(new_y, num_triplets)  # (one hot) （1500）
        return x, y

# 计算相似度
def call_similar(x):
    no_batch = int(x.shape[0] / (num_neg+2))  # each batch was consisted of 1 anchor ,1 positive , num_neg negative, so the number of batch
    similar = []
    for ep in range(no_batch):
        index = ep*(num_neg + 2)  # index是anchor的开始位置
        anchor = np.tile(x[index], (num_neg + 1, 1))  #(50,512)
        pos_neg = x[index+1: index + num_neg + 2]  #(50,512) 取anchor后面的postive和negative
        sim = batch_cosine_similarity(anchor, pos_neg)
        similar.extend(sim)
    return np.array(similar)


# 计算余弦距离 
def batch_cosine_similarity(x1,x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul,axis=1)

    #l1 = np.sum(np.multiply(x1, x1),axis=1)
    #l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s

# 评估模型
def eval_model(model,dataset,train_batch_size=BATCH_SIZE*TRIPLET_PER_BATCH,check_partial=False):
    x,y_true = create_test_data(dataset,check_partial)

    batch_size = x.shape[0]
    b = x[0]
    input_shape = (b.shape[0],b.shape[1],b.shape[2])
    
    test_epoch = int(len(y_true)/train_batch_size)  # 测试轮数  15
    
    embedding = None
    for ep in range(test_epoch):
        x_ = x[ep*train_batch_size: (ep + 1)*train_batch_size]  #(batch_size,299,40,1)
        embed = model.predict_on_batch(x_)  #(batch_size,512)
        if embedding is None:
            embedding = embed.copy()
        else:
            embedding = np.concatenate([embedding, embed], axis=0)  #(1440,512)
            
    y_pred = call_similar(embedding)  # （1440）
    nrof_pairs = min(len(y_pred), len(y_true))
    y_pred = y_pred[:nrof_pairs]
    y_true = y_true[:nrof_pairs]
    fm, tpr, acc, eer = evaluate(y_pred, y_true)
    return fm, tpr, acc, eer



    
if __name__ == "__main__":
    input_shape = (299,40,1)
    model= Att_DCNN.Att_DCNN().baseline_Model(input_shape)
    model_dir =  './../checkpoint/AttDCNN' # 模型保存目录
    model.load_weights(f'{model_dir}/best.h5', by_name='True')
    
    train_dir = "/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/train"
    train_dataset = data_catalog(train_dir)
    eval_model(model,train_dataset)