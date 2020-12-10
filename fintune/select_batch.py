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

alpha = 0.2
CANDIDATES_PER_BATCH = 640
HIST_TABLE_SIZE = 10

# 计算余弦距离 
def batch_cosine_similarity(x1,x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    # 方法1
    mul = np.multiply(x1, x2)
    s = np.sum(mul,axis=1)
    # 方法2 同方法1计算出的结果是一致的
    # s1 = []
    # for i in range(0,x1.shape[0]):
    #     sm = np.dot(x1[i], x2[i])/(np.linalg.norm(x1[i])*np.linalg.norm(x2[i]))# 计算余弦距离
    #     s1.append(sm)  
    return s

def matrix_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.dot(x1, x2.T)
    return mul


stack = []
def create_data_producer(unique_speakers, spk_utt_dict,candidates=CANDIDATES_PER_BATCH):
    producer = threading.Thread(target=addstack, args=(unique_speakers, spk_utt_dict,candidates))
    producer.setDaemon(True)
    producer.start()

def addstack(unique_speakers, spk_utt_dict,candidates=CANDIDATES_PER_BATCH):
    data_produce_step = 0
    while True:
        if len(stack) >= c.DATA_STACK_SIZE:
            sleep(0.01)
            continue

        orig_time = time()
        feature, labels = preprocess(unique_speakers, spk_utt_dict, candidates)
        #print("pre-process one batch data costs {0:.4f} s".format(time() - orig_time))
        stack.append((feature, labels))

        data_produce_step += 1
        if data_produce_step % 100 == 0:
            for spk in unique_speakers:
                np.random.shuffle(spk_utt_dict[spk])
def getbatch():
    while True:
        if len(stack)==0:
            continue
        return stack.pop(0)
    
    
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
    orig_time = time()
    embeds = model.predict_on_batch(features)
    print("forward process time {0:.3}s".format(time()-orig_time))
    
    if hist_embeds is None:
        hist_features = np.copy(features)
        hits_labels = np.copy(labels)
        hist_embeds = np.copy(embeds)
    else:
        if len(hist_labels) < hist_table_size*candidates:
            hist_features = np.concatenate((hist_features, features), axis=0)
            hist_labels = np.concatenate((hist_labels, labels), axis=0)
            hist_embeds = np.concatenate((hist_embeds, embeds), axis=0)
        else:
            hist_features[hist_index*candidates: (hist_index+1)*candidates] = features
            hist_labels[hist_index*candidates: (hist_index+1)*candidates] = labels
            hist_embeds[hist_index*candidates: (hist_index+1)*candidates] = embeds
            
    hist_index = (hist_index+1)%hist_table_size
    
    anchor_batch = []
    positive_batch = []
    negative_batch = []
    anchor_labs, positive_labs, negative_labs = [], [],  []
    
    orig_time = time()
    anh_speakers = np.random.choice(hist_labels, int(batch_size/2), replace=False)
    anchs_index_dict = {}
    inds_set = []
    for spk in anh_speakers:
        anhinds = np.argwhere(hist_labels==spk).flatten()
        anchs_index_dict[spk] = anhinds
        inds_set.extend(anhinds)
    inds_set = list(set(inds_set))
    
    speakers_embeds = hist_embeds[inds_set]
    sims = matrix_cosine_similarity(speakers_embeds, hist_embeds)
    
    print('beginning to select..........')
    
    for ii in range(int(batch_size/2)):   #每一轮找出两对triplet pairs
        while True:
            speaker = anh_speakers[ii]
            inds = anchs_index_dict[speaker]
            np.random.shuffle(inds)
            anchor_index = inds[0]
            pinds = []
            for jj in range(1,len(inds)):
                if (hist_features[anchor_index] == hist_features[inds[jj]]).all():
                    continue
                pinds.append(inds[jj])

            if len(pinds) >= 1:
                break
            
        sap = sims[ii][pinds]
        min_saps = heapq.nsmallest(2, sap)
        pos0_index = pinds[np.argwhere(sap == min_saps[0]).flatten()[0]]
        if len(pinds) > 1:
            pos1_index = pinds[np.argwhere(sap == min_saps[1]).flatten()[0]]
        else:
            pos1_index = pos0_index
        
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

    batch = np.concatenate([np.array(anchor_batch), np.array(positive_batch), np.array(negative_batch)], axis=0)
    labs = anchor_labs + positive_labs + negative_labs
    print("select best batch time {0:.3}s".format(time() - orig_time))
    return batch, np.array(labs)

if __name__ == "__main__":
    pass