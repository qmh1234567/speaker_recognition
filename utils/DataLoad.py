import os
import tensorflow as tf
from collections import Counter
import numpy as np
from progress.bar import Bar
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
from keras.layers import Flatten
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from keras import optimizers
import glob
import pickle
import argparse
from keras.utils import Sequence,np_utils
import python_speech_features as psf
import math

## 数据生成器类
class BaseSequence(Sequence):
    # x=paths y=labels
    def __init__(self,batch_size,x,y,num_class):
        self.x_y = np.concatenate((np.array(x).reshape(len(x),1),np.array(y).reshape(len(y),1)),axis=1)
        self.batch_size = batch_size
        self.num_class = num_class
    def __len__(self):
        return  math.ceil(len(self.x_y)/self.batch_size)

    def standard_normaliztion(self,x_array,epsilon=1e-12):
        return np.array([(x-np.mean(x))/max(np.std(x),epsilon) for x in x_array])

    def preprocess_x(self,batch_x,batch_y):
        X,Y = [],[]
        for index,x in enumerate(batch_x):
            try:
                with open(x,"rb") as f:
                    load_dict = pickle.load(f)
                    x = load_dict["LogMel_Features"]
                    x = self.standard_normaliztion(x)
                    x = x[:, :, np.newaxis]
                    X.append(x)
                    y = np_utils.to_categorical(batch_y[index],num_classes=self.num_class,dtype='int8')
                    Y.append(y)  
            except Exception as e:
                print(e)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y

    def __getitem__(self,index):
        batch_x = self.x_y[index*self.batch_size:(index+1)*self.batch_size,0]
        batch_y = self.x_y[index*self.batch_size:(index+1)*self.batch_size,1]

        batch_x,batch_y = self.preprocess_x(batch_x,batch_y)
        return batch_x,batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.x_y)


class DataLoad():
    
    def __init__(self):
        self.ANNONATION_FILE = './dataset/annonation.csv'  # 测试数据集表格
        self.ENROLL_NUMBER = 20  # 注册人数  推荐为测试数据集的一半作为注册人数
        self.ENROLL_FILE = './dataset/enrollment.csv' # 注册数据集表格

    # 标签的映射函数
    def Map_label_to_dict(self,labels):
        labels_to_id = {}
        i = 0
        for label in np.unique(labels):
            labels_to_id[label] = i
            i+= 1
        return labels_to_id

    # 数据流生成器
    def data_flow(self,train_pk_dir,batch_size,split_ratio=0.2):
        pickle_list  = [pickle for pickle in glob.iglob(train_pk_dir + "/*.pickle")]

        pickle_list.sort()

        audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in pickle_list]

        train_paths, val_paths, train_labels, val_labels = train_test_split(pickle_list, audio_labels,
                                                                                stratify=audio_labels, test_size=split_ratio, random_state=42)
        labels_to_id = self.Map_label_to_dict(audio_labels)

        train_labels = [labels_to_id[label] for label in train_labels]

        val_labels = [labels_to_id[label] for label in val_labels]

        # build sequence
        num_class = len(set(audio_labels))
        train_sequence = BaseSequence(batch_size,train_paths,train_labels,num_class)
        # batch_x,batch_y = train_sequence.__getitem__(6)
        # print(batch_x.shape)
        # print(batch_y.shape)
        # exit()
        val_sequence = BaseSequence(batch_size,val_paths,val_labels,num_class)
        return train_sequence,val_sequence,num_class

    #  打乱数据
    def shuffle_data(self,paths, labels,ismember=None):
        length = len(labels)
        shuffle_index = np.arange(0, length)
        shuffle_index = np.random.choice(shuffle_index, size=length, replace=False)
        paths = np.array(paths)[shuffle_index]
        labels = np.array(labels)[shuffle_index]
        if ismember is None:
            return paths, labels
        ismember = np.array(ismember)[shuffle_index]
        return paths,labels,ismember
        

    # 切分数据集
    def split_perspeaker_audios(self,audio_paths, audio_labels, split_ratio=0.1):
        val_paths, val_labels, train_paths, train_labels = [], [], [], []
        dict_count = Counter(audio_labels)
        
        for speaker in set(audio_labels):
            start = audio_labels.index(speaker)
            end = start + dict_count[speaker]
            # shuffle
            np.random.shuffle(audio_paths[start:end])
            # 前10%留作验证集
            for index in range(start, end):
                if index < start+dict_count[speaker]*split_ratio:
                    val_paths.append(audio_paths[index])
                    val_labels.append(speaker)
                else:
                    train_paths.append(audio_paths[index])
                    train_labels.append(speaker)
        return val_paths, val_labels, train_paths, train_labels
    
    
    # 构建测试数据集
    def createTestDataSet(self,test_pk_dir,target='SI',split_ratio=0.2):
        test_paths,val_paths = [],[]
        test_labels, val_labels = [], []
    
        audio_paths = [pickle for pickle in glob.iglob(test_pk_dir +"/*.pickle")]
        
        audio_paths.sort()
        
        audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in audio_paths]

        if target == 'SI':
            
            val_paths, val_labels, test_paths, test_labels = self.split_perspeaker_audios(
                    audio_paths, audio_labels, split_ratio)
            
            test_paths,test_labels = self.shuffle_data(test_paths,test_labels)
            # 测试集
            data_dict = {
                'FilePath': test_paths,
                'SpeakerID': test_labels,
            }
            data = pd.DataFrame(data_dict)
            data.to_csv(self.ANNONATION_FILE, index=0)
            print(f"wirte to {self.ANNONATION_FILE} succeed")
            # 注册数据集
            data_dict = {
                'FilePath': val_paths,
                'SpeakerID': val_labels
            }
            data = pd.DataFrame(data_dict)
            data.to_csv(self.ENROLL_FILE, index=0)
            print(f"wirte to {self.ENROLL_FILE} succeed")
            
        else:
            # if not os.path.exists(self.ANNONATION_FILE):

            dict_count = Counter(audio_labels)
            
            cut_index = 0
            
            # remove repeat and rank in order
            new_audio_labels = list(set(audio_labels))
            
            new_audio_labels.sort(key=audio_labels.index)
            
            # 从测试集中选出ENROLL_NUMBER个人作为注册说话人，[:cut_index]句话
            for index, speaker in enumerate(new_audio_labels):
                if index == self.ENROLL_NUMBER:
                    break
                else:
                    cut_index += dict_count[speaker]
                    
            # 用注册说话人构建测试数据集
            val_paths, val_labels, test_paths, test_labels = self.split_perspeaker_audios(audio_paths[:cut_index], audio_labels[:cut_index], split_ratio)
            ismember = np.ones(len(test_labels)).tolist()

            
            # 剩下的说话人句子拼接上非说话人的句子，组成评估集
            test_paths.extend(audio_paths[cut_index:])
            test_labels.extend(audio_labels[cut_index:])
            
            
            # is member
            ismember.extend(np.zeros(len(audio_labels[cut_index:])).tolist())
            

            # shuffle
            test_paths,test_labels,ismember = self.shuffle_data(test_paths,test_labels,ismember)
            
            # create annonation.csv
            # 测试集
            data_dict = {
                'FilePath': test_paths,
                'SpeakerID': test_labels,
                'Ismember': ismember,
            }
            data = pd.DataFrame(data_dict)
            data.to_csv(self.ANNONATION_FILE, index=0)
            print(f"wirte to {self.ANNONATION_FILE} succeed")
            # 注册数据集
            data_dict = {
                'FilePath': val_paths,
                'SpeakerID': val_labels
            }
            data = pd.DataFrame(data_dict)
            data.to_csv(self.ENROLL_FILE, index=0)
            print(f"wirte to {self.ENROLL_FILE} succeed")
            # else:
            #     data = pd.DataFrame(pd.read_csv(self.ANNONATION_FILE))
            #     train_paths = data['FilePath'].tolist()
            #     train_labels = data['SpeakerID'].tolist()
            #     ismember = data['Ismember'].tolist()

            #     data = pd.read_csv(self.ENROLL_FILE)
            #     val_paths = data['FilePath'].tolist()
            #     val_labels = data['SpeakerID'].tolist()

        test_dataset = (test_paths, test_labels)
        val_dataset = (val_paths, val_labels)
        
        print("len(train_paths)=", len(test_paths))
        print("len(val_paths)=", len(val_paths))
        print("len(audio_paths)=", len(audio_labels))
        print("len(set(train_labels))=", len(set(test_labels)))
        return test_dataset, val_dataset

    def standard_normaliztion(self,x_array,epsilon=1e-12):
        return np.array([(x-np.mean(x))/max(np.std(x),epsilon) for x in x_array])

    # 加载全部数据
    def load_all_data(self,dataset, typeName):
        (path, labels) = dataset
        X, Y = [], []
        bar = Bar('loading data', max=len(path), fill='#', suffix='%(percent)d%%')
        for index, audio in enumerate(path):
            bar.next()
            try:
                with open(audio, "rb") as f:    # 改
                    load_dict = pickle.load(f)
                    x = load_dict["LogMel_Features"]
                    x = self.standard_normaliztion(x)
                    x = x[:, :, np.newaxis]
                    X.append(x)
                    Y.append(labels[index])
            except Exception as e:
                print(e)
        bar.finish()
        return (np.array(X), np.array(Y))  # 这里的Y是说话人的名字，是标签

    # 加载数据集
    def loadDataset(self,fileName):
        path = []
        labels = []   
        for line in open(fileName):
            line = line.rstrip('\r\t\n ')
            utt, spk = line.split(' ')
            path.append(utt)
            labels.append(spk)
        return (path,labels)
            
       
        
        