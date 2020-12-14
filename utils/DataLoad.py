import pandas as pd
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

class DataLoad():
    
    def __init__(self):
        self.ANNONATION_FILE = './dataset/annonation.csv'  # 测试数据集表格
        self.ENROLL_NUMBER = 20  # 注册人数
        self.ENROLL_FILE = './dataset/enrollment.csv' # 注册数据集表格
    
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
    
    
    # 构建训练数据集
    def createTrainDataSet(self,train_pk_dir,split_ratio=0.2):
        train_paths,val_paths = [],[]
        train_labels, val_labels = [], []
        seed = 42
        np.random.seed(seed)
        
        speaker_pickle_files_list = [pickle for pickle in glob.iglob(train_pk_dir + "/*.pickle")]
        
        audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in speaker_pickle_files_list]
             
        train_paths, val_paths, train_labels, val_labels = train_test_split(speaker_pickle_files_list, audio_labels,
                                                                                stratify=audio_labels, test_size=split_ratio, random_state=42)
        train_dataset = (train_paths, train_labels)
        val_dataset = (val_paths, val_labels)
        
        print("len(train_paths)=", len(train_paths))
        print("len(val_paths)=", len(val_paths))
        print("len(audio_paths)=", len(audio_labels))
        print("len(set(train_labels))=", len(set(train_labels)))
        return train_dataset, val_dataset

    
    # 构建测试数据集
    def createTestDataSet(self,test_pk_dir,target='SI',split_ratio=0.2):
        train_paths,val_paths = [],[]
        train_labels, val_labels = [], []
    
        audio_paths = [pickle for pickle in glob.iglob(test_pk_dir +"/*.pickle")]
        
        audio_paths.sort()
        
        audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in audio_paths]
        
        if target == 'SI':
            
            val_paths, val_labels, train_paths, train_labels = self.split_perspeaker_audios(
                    audio_paths, audio_labels, split_ratio)
        else:
            if not os.path.exists(self.ANNONATION_FILE):

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
                val_paths, val_labels, train_paths, train_labels = self.split_perspeaker_audios(audio_paths[:cut_index], audio_labels[:cut_index], split_ratio)
                ismember = np.ones(len(train_labels)).tolist()
                
                # 剩下的就是非说话人 non-speaker
                train_paths.extend(audio_paths[cut_index:])
                train_labels.extend(audio_labels[cut_index:])
                
                # is member
                ismember.extend(np.zeros(len(audio_labels[cut_index:])).tolist())
                
                # shuffle
                train_paths,train_labels,ismember = self.shuffle_data(train_paths,train_labels,ismember)
                
                # create annonation.csv
                # 测试集
                data_dict = {
                    'FilePath': train_paths,
                    'SpeakerID': train_labels,
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
            else:
                data = pd.DataFrame(pd.read_csv(self.ANNONATION_FILE))
                train_paths = data['FilePath'].tolist()
                train_labels = data['SpeakerID'].tolist()
                ismember = data['Ismember'].tolist()

                data = pd.read_csv(self.ENROLL_FILE)
                val_paths = data['FilePath'].tolist()
                val_labels = data['SpeakerID'].tolist()

        train_dataset = (train_paths, train_labels)
        val_dataset = (val_paths, val_labels)
        
        print("len(train_paths)=", len(train_paths))
        print("len(val_paths)=", len(val_paths))
        print("len(audio_paths)=", len(audio_labels))
        print("len(set(train_labels))=", len(set(train_labels)))
        return train_dataset, val_dataset

    # 说话人标签映射为id
    def Map_label_to_dict(self,labels):
        labels_to_id = {}
        i = 0
        for label in np.unique(labels):
            labels_to_id[label] = i
            i += 1
        return labels_to_id
    

    # 加载验证数据集
    def load_validation_data(self,dataset, labels_to_id, num_class):
        (path, labels) = dataset
        path, labels = self.shuffle_data(path, labels,None)
        X, Y = [], []
        bar = Bar('loading data', max=len(labels),fill='#', suffix='%(percent)d%%')
        for index, pk in enumerate(path):
            bar.next()
            try:
                with open(pk, "rb") as f:
                    load_dict = pickle.load(f)
                    x = load_dict["LogMel_Features"]
                    x = x[:, :, np.newaxis]
                    X.append(x)
                    Y.append(labels_to_id[labels[index]])
            except Exception as e:
                print(e)
        X = np.array(X)
        Y = np.eye(num_class)[Y]
        bar.finish()
        return (X, Y)

    # 加载全部数据
    def load_all_data(self,dataset, typeName):
        (path, labels) = dataset
        X, Y = [], []
        bar = Bar('loading data', max=len(path), fill='#', suffix='%(percent)d%%')
        for index, audio in enumerate(path):
            bar.next()
            try:
                with open(audio, "rb") as f:
                    load_dict = pickle.load(f)
                    x = load_dict["LogMel_Features"]
                    x = x[:, :, np.newaxis]
                    X.append(x)
                    Y.append(labels[index])
            except Exception as e:
                print(e)
        bar.finish()
        return (np.array(X), np.array(Y))  # 这里的Y是说话人的名字，是标签

    # 加载每个批次的数据
    def load_each_batch(self,dataset, labels_to_id, batch_start, batch_end, num_class):
        (paths, labels) = dataset
        X, Y = [], []
        for i in range(batch_start, batch_end):
            try:
                with open(paths[i], "rb") as f:
                    load_dict = pickle.load(f)
                    x = load_dict["LogMel_Features"]
                    x = x[:, :, np.newaxis]
                    X.append(x)
                    Y.append(labels_to_id[labels[i]])
            except Exception as e:
                print(e)
        X = np.array(X)
        Y = np.eye(num_class)[Y]   
        return X, Y    # 这里的Y是one hot向量

    # 数据生成器
    def Batch_generator(self,dataset, labels_to_id, batch_size, num_class):
        (paths, labels) = dataset
        length = len(labels)
        while True:
            # shuffle
            paths, labels = self.shuffle_data(paths, labels)
            shuffle_dataset = (paths, labels)
            batch_start = 0
            batch_end = batch_size
            while batch_end < length:
                X, Y = self.load_each_batch(
                    shuffle_dataset, labels_to_id, batch_start, batch_end, num_class)
                yield (X, Y)
                batch_start += batch_size
                batch_end += batch_size
                
       
        
        