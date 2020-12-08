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

import usedModels.Deep_Speaker as Deep_Speaker
import usedModels.VggVox as VggVox
import usedModels.SE_ResNet as SE_ResNet
import usedModels.Att_DCNN as Att_DCNN

import utils.DataLoad as DataLoad
import utils.Util as Util
import utils.LossHistory as LossHistory

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)

MODEL_DIR = './checkpoint/' # 模型保存目录
LEARN_RATE = 0.01 # 学习率
BATCH_SIZE = 32
EPOCHS = 300# 训练轮次

# 选择模型
def createModel(modelName,input_shape=(299,40,1)):
    if modelName == "deepSpk":
        model = Deep_Speaker.DeepSpeaker().deep_speaker_model(input_shape)
    elif modelName== "VggVox":
        model = VggVox.VggVox().res_34(input_shape)
    elif modelName== "SEResNet":
        model = SE_ResNet.SE_ResNet().se_resNet(input_shape)
    elif modelName == "AttDCNN":
        model = Att_DCNN.Att_DCNN().baseline_Model(input_shape)
    # print(model.summary())
    return model

# add softmax layer
def addSoftmax(model,nclass):
    x = model.output
    x = Dense(nclass, activation='softmax', name=f'softmax')(x)
    model = Model(model.input, x)
    return model


def train(model,dataLoad,hparams):
    
    model_dir = MODEL_DIR + hparams.model_name
    
    if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            
    train_dataset, val_dataset = dataLoad.createTrainDataSet(hparams.train_pk_dir)
    
    nclass = len(set(train_dataset[1]))
    
    print("nclass = ",nclass)
    
    labels_to_id = dataLoad.Map_label_to_dict(train_dataset[1])
    
    model = addSoftmax(model,nclass) 
    
    # print(model.summary())
    
    
    # train model
    
    history = LossHistory.LossHistory()
    
    sgd = optimizers.SGD(lr=LEARN_RATE,momentum=0.9) #TIMIT libri-seresnet
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                    metrics=['accuracy'])
    
    
    model.fit_generator(dataLoad.Batch_generator(train_dataset, labels_to_id, BATCH_SIZE, nclass),
                        steps_per_epoch=len(train_dataset[0])//BATCH_SIZE, epochs=EPOCHS,
                        validation_data=dataLoad.load_validation_data(
                            val_dataset, labels_to_id, nclass),
                        validation_steps=len(val_dataset[0])//BATCH_SIZE,
                        callbacks=[
                            ModelCheckpoint(f'{model_dir}/best.h5',
                                monitor='val_loss', save_best_only=True, mode='min'),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=50, mode='min'),
                            # EarlyStopping(monitor='val_loss', patience=10),
                            history,
    ])
    
    # 绘制loss曲线
    history.loss_plot('epoch')
    

def test(model,dataLoad,util,hparams):
    
    # print(model.summary())
    
    test_dataset, enroll_dataset = dataLoad.createTestDataSet(
            hparams.test_pk_dir,target=hparams.target, split_ratio=0.5)
    
    labels_to_id = dataLoad.Map_label_to_dict(labels=enroll_dataset[1])
    
    # load weights
    model_dir = MODEL_DIR + hparams.model_name
    model.load_weights(f'{model_dir}/best.h5', by_name='True')
        
    # load all data
    print("loading data...") 
    (enroll_x, enroll_y) = dataLoad.load_all_data(enroll_dataset, 'enroll')
    (test_x, test_y) = dataLoad.load_all_data(test_dataset, 'test')
    
    # 预测  主要获取说话人嵌入
    enroll_pre = np.squeeze(model.predict(enroll_x))
    test_pre = np.squeeze(model.predict(test_x))
    
    # 计算余弦距离
    distances = util.caculate_distance(enroll_dataset, enroll_pre, test_pre)
    
    if hparams.target=="SI":
        # speaker identification
        test_y_pre = util.speaker_identification(
            enroll_dataset, distances, enroll_y)
        
        # compute result
        result = util.compute_result(test_y_pre, test_y)
        
        score = sum(result)/len(result)
        print(f"score={score}")
    else:
        df = pd.read_csv(dataLoad.ANNONATION_FILE)

        ismember_true = list(map(int, df['Ismember']))
        # np.save('./npys/perfect_noELU.npy',distances)
        ismember_pre = util.speaker_verification(distances, ismember_true)
        
        # compute result
        result = util.compute_result(ismember_pre, ismember_true)
        
        score = sum(result)/len(result)
        print(f"score={score}")

def main(hparams):    
        
    model = createModel(hparams.model_name)

    dataLoad = DataLoad.DataLoad()
    
    if hparams.stage.startswith('train'):
        
        train(model,dataLoad,hparams)
        
    else:
        util = Util.Util()
        
        test(model,dataLoad,util,hparams)
    


if __name__ == "__main__":
    
    # nohup python run.py --stage="train" --model_name="deepSpk" --target="SI" > output.out &
    # python run.py --stage="test" --model_name="deepSpk" --target="SV"
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--stage",type=str,required=True,help="train or test",choices=["train","test"])

    parser.add_argument("--model_name",type=str,required=True,help="model's name",choices=["deepSpk","VggVox","SEResNet","AttDCNN"])
    
    parser.add_argument("--target",type=str,required=True,help="SV or SI ,which is used in test stage",choices=["SV","SI"])
    
    parser.add_argument("--train_pk_dir",type=str,help="train pickle dir",default="/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/train")
    
    parser.add_argument("--test_pk_dir",type=str,help="test pickle dir",default="/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/test")
    
    args = parser.parse_args()
    
    main(args)