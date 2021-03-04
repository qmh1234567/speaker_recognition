# 选择最优的batch
# 对于每个speaker,选择两个最不相似的positives和两个最相似的negatives,组成2对pairs

import pandas as pd
import random
import numpy as np
from triplet_loss import deep_speaker_loss
from random_batch import data_catalog
import heapq
import threading
from time import time, sleep
import os
import pickle
import tensorflow as tf
import sys
sys.path.append("../") 
import usedModels.Att_DCNN as Att_DCNN

alpha = 0.2
CANDIDATES_PER_BATCH = 32  # 每个batch的候选人有x句话，x/2个说话人   # SECNN要改成64
HIST_TABLE_SIZE = 10
DATA_STACK_SIZE = 10    # stack大小超过10就要休眠0.01秒


# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)


def matrix_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.dot(x1, x2.T)
    return mul


# 获取特征和标签
spk_utt_index = {}
def preprocess(unique_speakers,spk_utt_dict,candidates=CANDIDATES_PER_BATCH):
    files = []
    flag = False if len(unique_speakers) > candidates/2 else True
    # 选candidates/2个人出来
    speakers = np.random.choice(unique_speakers,size=int(candidates/2),replace=flag)  
    # 每个人选两个音频
    for speaker in speakers:
        index = 0
        ll = len(spk_utt_dict[speaker])
        if speaker in spk_utt_index:
            index = spk_utt_index[speaker] % ll
        files.append(spk_utt_dict[speaker][index])
        files.append(spk_utt_dict[speaker][(index+1)%ll])
        spk_utt_index[speaker] = (index+2)%ll   # 存放还没有被选过的音频起始下标
        
    x,y = to_inputs(files)   # x.shape=(640,299,40,1) y.shape=(640)  320人，每人两个音频
    return x,y

def standard_normaliztion(x_array,epsilon=1e-12):
        return np.array([(x-np.mean(x))/max(np.std(x),epsilon) for x in x_array])

# 产生输入
def to_inputs(files):
    new_x = []
    y = []
    for filename in files:
        with open(filename,"rb") as f:
            load_dict = pickle.load(f)
            x = load_dict["LogMel_Features"]
            x = standard_normaliztion(x)
            x = x[:, :, np.newaxis]
            new_x.append(x)
        y.append(os.path.basename(filename).split("_")[0])
    x = np.array(new_x) 
    y = np.array(y)
    return x, y


# 多线程加载数据
stack = []
def create_data_producer(unique_speakers, spk_utt_dict,candidates=CANDIDATES_PER_BATCH):
    # 定义一个线程
    producer = threading.Thread(target=addstack, args=(unique_speakers, spk_utt_dict,candidates))
    producer.setDaemon(True)  # 守护线程
    producer.start()

# 线程需要做的任务 不断向stack里面添加输入, 每10组休眠0.01秒，然后继续添加
def addstack(unique_speakers, spk_utt_dict,candidates=CANDIDATES_PER_BATCH):
    data_produce_step = 0
    while True:
        if len(stack) >= DATA_STACK_SIZE: 
            sleep(0.1)  
            continue
    
        feature, labels = preprocess(unique_speakers, spk_utt_dict, candidates)
        stack.append((feature, labels))
        
        # 每100次打乱一下每个说话人的所有音频
        data_produce_step += 1
        if data_produce_step % 100 == 0:
            for spk in unique_speakers:
                np.random.shuffle(spk_utt_dict[spk])


## 弹出stack里面的输入，每次弹出一组，得到一个batch
def getbatch():
    while True:
        if len(stack)==0:
            sleep(0.01)
            continue
        else:
            # print("len(stack)",len(stack))
            return stack.pop(0)


# 选择最好的batch,从候选集的x/2个说话人中取16个说话人，每个人3句话
hist_embeds = None
hist_labels = None
hist_features = None
hist_index = 0
hist_table_size = HIST_TABLE_SIZE
def best_batch(model,batch_size,candidates=CANDIDATES_PER_BATCH):
    orig_time = time()
    global hist_embeds, hist_features, hist_labels, hist_index, hist_table_size
    features,labels = getbatch()
    print("get batch time {0:.3}s".format(time() - orig_time))
    
    # orig_time = time()
    embeds = model.predict_on_batch(features)  # (640,512)
    # print("forward process time {0:.3}s".format(time()-orig_time))
    if hist_embeds is None:
        hist_features = np.copy(features)  
        hist_labels = np.copy(labels)  #(640)
        hist_embeds = np.copy(embeds)  #(640,512)
    else:
        if len(hist_labels) < hist_table_size*candidates:  # 6400 小于6400则进行追加
            hist_features = np.concatenate((hist_features, features), axis=0)
            hist_labels = np.concatenate((hist_labels, labels), axis=0)
            hist_embeds = np.concatenate((hist_embeds, embeds), axis=0)
        else:  #>=6400 则进行替换
            hist_features[hist_index*candidates: (hist_index+1)*candidates] = features
            hist_labels[hist_index*candidates: (hist_index+1)*candidates] = labels
            hist_embeds[hist_index*candidates: (hist_index+1)*candidates] = embeds   
            
    hist_index = (hist_index+1)%hist_table_size  # 下标循环取值

    anchor_batch = []
    positive_batch = []
    negative_batch = []
    anchor_labs, positive_labs, negative_labs = [], [],  []
    
    anh_speakers = np.random.choice(hist_labels, int(batch_size/2), replace=False)  # 从labels中随机选batch_size/2个标签
    anchs_index_dict = {} # key=说话人标签spk value=该说话人在hist_labels中的所有下标
    inds_set = []
    for spk in anh_speakers:
        anhinds = np.argwhere(hist_labels==spk).flatten()   # 返回spk在hist_labels中的所有下标
        anchs_index_dict[spk] = anhinds  
        inds_set.extend(anhinds)  # 将这些说话人的下标存储起来
    inds_set = list(set(inds_set))  # 其中可能有相同的说话人,故需要对下标去重
    speakers_embeds = hist_embeds[inds_set]  # 选出不同说话人的嵌入
    sims = matrix_cosine_similarity(speakers_embeds, hist_embeds) # 不同说话人的嵌入矩阵与当前所有说话人的嵌入矩阵做矩阵乘法  (batch_size/2，len(hist_labels))

    # print("predict time {0:.3}s".format(time()-orig_time))
    # print('beginning to select..........')
    
    
    # orig_time = time()
    for ii in range(int(batch_size/2)):   #每一轮找出两对triplet pairs
        while True:
            speaker = anh_speakers[ii]
            inds = anchs_index_dict[speaker]
            np.random.shuffle(inds)  # 打乱这些音频
            anchor_index = inds[0]  # 选择第一个作为anchor
            pinds = []  # 剩下的音频都是positive
            for jj in range(1,len(inds)):
                if (hist_features[anchor_index] == hist_features[inds[jj]]).all():
                    continue
                pinds.append(inds[jj])

            if len(pinds) >= 1:
                break

        sap = sims[ii][pinds]  # 这里的值有点大了，超过了100
        min_saps = heapq.nsmallest(2, sap)  # 从sap中返回最小的两个值
        
        # 正样本下标
        pos0_index = pinds[np.argwhere(sap == min_saps[0]).flatten()[0]]
        if len(pinds) > 1:
            pos1_index = pinds[np.argwhere(sap == min_saps[1]).flatten()[0]]
        else:
            pos1_index = pos0_index
            
        #负样本下标
        ninds = np.argwhere(hist_labels != speaker).flatten()
        san = sims[ii][ninds]
        max_sans = heapq.nlargest(2, san)
        neg0_index = ninds[np.argwhere(san == max_sans[0]).flatten()[0]]
        neg1_index = ninds[np.argwhere(san == max_sans[1]).flatten()[0]]

        anchor_batch.append(hist_features[anchor_index]);  anchor_batch.append(hist_features[anchor_index])
        positive_batch.append(hist_features[pos0_index]);  positive_batch.append(hist_features[pos1_index])
        negative_batch.append(hist_features[neg0_index]);  negative_batch.append(hist_features[neg1_index])

        anchor_labs.append(hist_labels[anchor_index]);  anchor_labs.append(hist_labels[anchor_index])
        positive_labs.append(hist_labels[pos0_index]);  positive_labs.append(hist_labels[pos1_index])
        negative_labs.append(hist_labels[neg0_index]);  negative_labs.append(hist_labels[neg1_index])
    
    batch = np.concatenate([np.array(anchor_batch), np.array(positive_batch), np.array(negative_batch)], axis=0)  #(96,299,40,1)
    labs = anchor_labs + positive_labs + negative_labs
    # print("select best batch time {0:.3}s".format(time() - orig_time))
    return batch, np.array(labs)




if __name__ == "__main__":
    dataset_dir = "/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/train/"
    dataset = data_catalog(dataset_dir)
    
    unique_speakers = dataset['speaker_id'].unique()
    labels = dataset['speaker_id'].tolist()
    files = dataset['filename'].tolist()
    
    spk_utt_dict = {}  # key=speaker_id  value: 当前说话人的所有音频
    for i in range(len(unique_speakers)):
        spk_utt_dict[unique_speakers[i]] = []

    for i in range(len(labels)):
        spk_utt_dict[labels[i]].append(files[i])

    input_shape = (299,40,1)
    model= Att_DCNN.Att_DCNN().proposed_model(input_shape)
    model_dir =  './../checkpoint/AttDCNN/timit' # 模型保存目录
    model.load_weights(f'{model_dir}/best.h5', by_name='True')
    
    
    create_data_producer(unique_speakers,spk_utt_dict)

    x,y = best_batch(model,32)
    print(x.shape)
    print(y.shape)
    print(y)
    print(np.random.uniform(size=(x.shape[0], 1)))
    
    
    
    
    
    
    
    
    
    
    