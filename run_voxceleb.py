#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: run.py
# __time__: 2019:06:27:20:53

import pandas as pd
import constants as c
import os
import tensorflow as tf
from collections import Counter
import numpy as np
from progress.bar import Bar
import models
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import sys
import keras.backend.tensorflow_backend as KTF
from keras.layers import Flatten
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from keras import optimizers
import glob
import pickle

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)


def shuffle_data(paths,labels):
    length = len(labels)
    shuffle_index = np.arange(0,length)
    shuffle_index = np.random.choice(shuffle_index, size=length, replace=False)
    paths = np.array(paths)[shuffle_index]
    labels = np.array(labels)[shuffle_index]
    return paths,labels

def split_Voxceleb_SIset(SI_text_path):
    train_paths, val_paths, test_paths= [], [], []
    train_labels, val_labels, test_labels = [],[],[]
    speaker_test = os.listdir(c.TEST_SET.replace("pickle",'wav'))
    with open(SI_text_path,'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            set_type = line.split(' ')[0]
            audio_path = line.split(' ')[1].replace("/","_").replace("wav","pickle")
            speaker = audio_path.split('_')[0]
            if speaker in speaker_test:
                dir_name = c.TEST_SET
            else:
                dir_name = c.TRAIN_DEV_SET
            if set_type == '1':
                train_paths.append(os.path.join(dir_name,audio_path))
                train_labels.append(speaker)
            elif set_type == '2':
                val_paths.append(os.path.join(dir_name,audio_path))
                val_labels.append(speaker)
            else:
                test_paths.append(os.path.join(dir_name,audio_path))
                test_labels.append(speaker)
        train_dataset = (train_paths,train_labels)
        val_dataset = (val_paths,val_labels)
        test_dataset = (test_paths,test_labels)
        print("len(train_paths)=", len(train_paths))
        print("len(val_paths)=", len(val_paths))
        print("len(test_paths)=",len(test_paths))
        return train_dataset,val_dataset,test_dataset

def split_Voxceleb_SVset(split_ratio,nclass):
    pickle_files_list = [pickle for pickle in glob.iglob(c.TRAIN_DEV_SET+"/*.pickle")]
    pickle_files_list.sort()
    audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in pickle_files_list]
    # cut speakers
    cut_index = 0
    speaker_count = Counter(audio_labels)
    sort_audio_labels = list(set(audio_labels))
    sort_audio_labels.sort(key=audio_labels.index)
    for index,speaker in enumerate(sort_audio_labels):
        if index == nclass:
            break
        else:
            cut_index += speaker_count[speaker]
    train_paths, val_paths, train_labels, val_labels = train_test_split(pickle_files_list[:cut_index],audio_labels[:cut_index],stratify=audio_labels[:cut_index],test_size = split_ratio,random_state=42)
    train_dataset = (train_paths,train_labels)
    val_dataset = (val_paths,val_labels)
    print("len(train_paths)=", len(train_paths))
    print("len(val_paths)=", len(val_paths))
    print("len(audio_paths)=", len(audio_labels))
    print("len(set(train_labels))=", len(set(train_labels)))
    return train_dataset,val_dataset



def Map_label_to_dict(labels):
    labels_to_id = {}
    i = 0
    for label in np.unique(labels):
        labels_to_id[label] = i
        i +=1
    return labels_to_id 


def load_validation_data(dataset,labels_to_id,num_class):
    (path,labels) = dataset
    path,labels = shuffle_data(path,labels)
    X,Y = [],[]
    bar = Bar('loading data',max=len(labels),fill='#',suffix='%(percent)d%%')
    for index,pk in enumerate(path):
        bar.next()
        try:
            with open(pk,"rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:,:,np.newaxis]
                X.append(x)
                Y.append(labels_to_id[labels[index]])
        except Exception as e:
            print(e)
    X = np.array(X)
    Y = np.eye(num_class)[Y]
    bar.finish()
    return (X,Y)

def load_all_data(dataset,typeName):
    (path,labels) = dataset
    X,Y = [],[]
    bar = Bar('loading data',max=len(path),fill='#',suffix='%(percent)d%%')
    for index,audio in enumerate(path):
        bar.next()
        try:
            with open(audio,"rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:,:,np.newaxis]
                X.append(x)
                Y.append(labels[index])
        except Exception as e:
            print(e)
    bar.finish()
    return (np.array(X),np.array(Y))

def load_each_batch(dataset,labels_to_id,batch_start,batch_end,num_class):
    (paths,labels) = dataset
    X,Y= [],[]
    for i in range(batch_start,batch_end):
        try:
            with open(paths[i],"rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:,:,np.newaxis]
                X.append(x)
                Y.append(labels_to_id[labels[i]])
        except Exception as e:
            print(e)
    X = np.array(X)
    Y = np.eye(num_class)[Y]
    return X,Y

def Batch_generator(dataset,labels_to_id,batch_size,num_class):
    (paths,labels) = dataset
    length = len(labels)
    while True:
        # shuffle
        paths,labels = shuffle_data(paths,labels)
        shuffle_dataset = (paths,labels)
        batch_start = 0
        batch_end = batch_size
        while batch_end < length:
            X,Y = load_each_batch(shuffle_dataset,labels_to_id,batch_start,batch_end,num_class)
            yield (X,Y)
            batch_start += batch_size
            batch_end += batch_size

def caculate_distance(enroll_dataset,enroll_pre,test_pre):
    print("enroll_pre.shape=",enroll_pre.shape)
    dict_count = Counter(enroll_dataset[1])
    print(dict_count)
    # each person get a enroll_pre
    speakers_pre = []
    # remove repeat
    enroll_speakers = list(set(enroll_dataset[1]))
    enroll_speakers.sort(key=enroll_dataset[1].index)
    for speaker in enroll_speakers:
        start = enroll_dataset[1].index(speaker)
        speaker_pre = enroll_pre[start:dict_count[speaker]+start]
        speakers_pre.append(np.mean(speaker_pre,axis=0))

    enroll_pre = np.array(speakers_pre)
    print("new_enroll_pre.shape=",enroll_pre.shape)
    # caculate distance
    distances = []
    print("test_pre.shape=",test_pre.shape)
    for i in range(enroll_pre.shape[0]):
        temp = []
        for j in range(test_pre.shape[0]):
            x = enroll_pre[i]
            y = test_pre[j]
            s = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
            temp.append(s)
        distances.append(temp)
    distances = np.array(distances)
    print("distances.shape=",distances.shape)
    return distances

def speaker_identification(enroll_dataset,distances,enroll_y):
    #  remove repeat
    new_enroll_y = list(set(enroll_y))
    new_enroll_y.sort(key=list(enroll_y).index)
    #  return the index of max distance of each sentence
    socre_index = distances.argmax(axis=0)
    y_pre = []
    for i in socre_index:
        y_pre.append(new_enroll_y[i])
    return y_pre

def compute_result(y_pre,y_true):  
    result= []
    for index,x in enumerate(y_pre):
        result.append(1 if x== y_true[index] else 0 )
    return result


def evaluate_metrics(y_true,y_pre):
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pre,pos_label=1)
    auc = metrics.auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,color = 'green',label = 'ROC')
    plt.plot(np.arange(1,0,-0.01),np.arange(0,1,0.01))
    plt.legend()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'ROC curve, AUC score={auc}')
    plt.show()

    threshold_index = np.argmin(abs(1-tpr - fpr))  
    threshold = thresholds[threshold_index]
    eer = ((1-tpr)[threshold_index]+fpr[threshold_index])/2
    print(eer)
    auc_score = metrics.roc_auc_score(y_true,y_pre,average='macro')

    y_pro =[ 1 if x > threshold else 0 for x in y_pre]
    acc = metrics.accuracy_score(y_true,y_pro)
    prauc = metrics.average_precision_score(y_true,y_pro,average='macro')
    return y_pro,eer,prauc,acc,auc_score
    

def speaker_verification(distances,ismember_true):
    score_index = distances.argmax(axis=0)
    distance_max = distances.max(axis=0)
    distance_max = (distance_max + 1 ) /2
    y_pro,eer,prauc,acc,auc_score = evaluate_metrics(ismember_true,distance_max)
    print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')
    return y_pro
    
def voxceleb_predict_sample(model,wav):
    info = os.path.splitext(wav)[0].split('/')
    audio_name = '_'.join(info)
    with open(f'{c.TEST_SET}/{audio_name}.pickle','rb') as f:
        load_dict = pickle.load(f)
        x = load_dict["LogMel_Features"]
        x = x[np.newaxis,:,:,np.newaxis]
        pre = np.squeeze(model.predict(x))
    return pre

def voxceleb_verification(model,test_txt):
    y_true, y_pre= [],[]
    with open(test_txt,'r') as f:
        lines = f.read().split('\n')
        bar = Bar('predicting....',max=len(lines),fill='#',suffix='%(percent)d%%')
        for line in lines:
            bar.next()
            target, enroll_wav , test_wav = line.split(' ')
            y_true.append(int(target))
            enroll_pre = voxceleb_predict_sample(model,enroll_wav)
            test_pre = voxceleb_predict_sample(model,test_wav)
            score = np.dot(enroll_pre,test_pre)/(np.linalg.norm(enroll_pre)*np.linalg.norm(test_pre))
            score = (score+1)/2
            y_pre.append(score)
        bar.finish()
    y_pro,eer,prauc,acc,auc_score = evaluate_metrics(y_true,y_pre)
    print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')


def voxceleb_identification(model,test_txt):
    pass



def main(typeName):
    # model = models.Deep_speaker_model(c.INPUT_SHPE)
    model = models.SE_ResNet(c.INPUT_SHPE)
    # model = models.RWCNN_LSTM(c.INPUT_SHPE)
    if c.TARGET == 'SI':
        train_dataset,val_dataset,test_dataset = split_Voxceleb_SIset(c.TEST_TEXT_SI)
        nclass = len(set(train_dataset[1]))
        print("nclass = ",nclass)
    else:
        nclass = 400
        train_dataset,val_dataset = split_Voxceleb_SVset(0.1,nclass)
        nclass = len(set(train_dataset[1]))
        print("nclass = ",nclass)

    if typeName.startswith('train'):
        if not os.path.exists(c.MODEL_DIR):
            os.mkdir(c.MODEL_DIR)
    
        labels_to_id = Map_label_to_dict(labels=train_dataset[1])
        # add softmax layer
        x = model.output
        x = Dense(nclass,activation='softmax',name=f'softmax')(x)
        model = Model(model.input,x)  
        print(model.summary())
        # train model 
        sgd = optimizers.SGD(lr=c.LEARN_RATE,momentum=0.9)
        model.compile(loss='categorical_crossentropy',optimizer = sgd,
        metrics=['accuracy'])
        model.fit_generator(Batch_generator(train_dataset,labels_to_id,c.BATCH_SIZE,nclass),
        steps_per_epoch=len(train_dataset[0])//c.BATCH_SIZE,epochs = 50,
        validation_data=load_validation_data(val_dataset,labels_to_id,nclass),
        validation_steps=len(val_dataset[0])//c.BATCH_SIZE,
        callbacks=[
            ModelCheckpoint(f'{c.MODEL_DIR}/best1.h5',monitor='val_loss',save_best_only=True,mode='min'),
            ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,mode='min'),
            EarlyStopping(monitor='val_loss',patience=10),
        ])
    else:
        # load weights
        model.load_weights(f'{c.MODEL_DIR}/best1_vox.h5',by_name='True')
        if c.TARGET == 'SI':
            test_x,test_y = test_dataset
            enroll_x,enroll_y = val_dataset
            enroll_pre = np.squeeze(model.predict(enroll_x))
            test_pre = np.squeeze(model.predict(test_x))
            distances = caculate_distance(val_dataset,enroll_pre,test_pre)
            # speaker identification
            test_y_pre = speaker_identification(val_dataset,distances,enroll_y)
            # compute result
            result = compute_result(test_y_pre,test_y)
            score = sum(result)/len(result)
            print(f"score={score}")
        elif c.TARGET == 'SV':
            voxceleb_verification(model,c.TEST_TEXT_SV)
        else:
            print("you should set the c.TARGET to SI and SV")
            exit(-1)
        

        
                
                
           
          
        



if __name__ == "__main__":
    # if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test']:
    #     print('Usage: python run.py [run_type]\n',
    #           '[run_type]: train | test')
    #     exit()
    mode = sys.argv[1]
    main(mode)
 