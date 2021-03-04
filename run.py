#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: run.py
# __time__: 2019:06:27:20:53

import pandas as pd
import os
import tensorflow as tf
from collections import Counter
import numpy as np
import keras
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

import finetune.eval_metrics as eval_metrics

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)

MODEL_DIR = './checkpoint/' # 模型保存目录
LEARN_RATE = 0.01 # 学习率
BATCH_SIZE = 32
EPOCHS = 300 # 训练轮次

# 选择模型
def createModel(modelName,input_shape=(299,40,1)):
    if modelName == "deepSpk":
        model = Deep_Speaker.DeepSpeaker().deep_speaker_model(input_shape)
    elif modelName== "VggVox":
        model = VggVox.VggVox().res_34(input_shape)
    elif modelName== "SEResNet":
        model = SE_ResNet.SE_ResNet().se_resNet(input_shape)
    elif modelName == "AttDCNN":
        model = Att_DCNN.Att_DCNN().proposed_model(input_shape)
    print(model.summary())
    return model

# add softmax layer
def addSoftmax(model,nclass):
    x = model.output
    x = Dense(nclass, activation='softmax', name=f'softmax')(x)
    model = Model(model.input, x)
    return model


def train(model,dataLoad,hparams):
    
    dataSetName = hparams.train_pk_dir.split("/")[-3].split("_")[0].lower()
    
    model_dir = os.path.join(MODEL_DIR + hparams.model_name,dataSetName)
        
    if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            
    # batch sequence
    train_sequence,val_sequence,nclass = dataLoad.data_flow(hparams.train_pk_dir,BATCH_SIZE)
    
    
    model = addSoftmax(model,nclass) 

    # train model
    history = LossHistory.LossHistory()
    
    sgd = optimizers.SGD(lr=LEARN_RATE,momentum=0.9) #TIMIT libri-seresnet
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                    metrics=['accuracy'])
    
    model.fit_generator(train_sequence,
                        steps_per_epoch=len(train_sequence), epochs=EPOCHS,
                        validation_data=val_sequence,
                        validation_steps=len(val_sequence),
                        callbacks=[
                            ModelCheckpoint(f'{model_dir}/best1.h5',
                                monitor='val_loss', save_best_only=True, mode='min'),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=10, mode='min'),
                            EarlyStopping(monitor='val_loss', patience=10),
                            history,
    ])
    
    # 绘制loss曲线
    history.loss_plot('epoch',dataSetName,hparams.model_name)
    

def test(model,dataLoad,util,hparams):
    
    # print(model.summary())
    
    eval_dataset, enroll_dataset = dataLoad.createTestDataSet(
            hparams.test_pk_dir,target=hparams.target, split_ratio=0.5)
    
    dataSetName = hparams.train_pk_dir.split("/")[-3].split("_")[0].lower()
    
    dataSetDir = os.path.join("./dataset",dataSetName)
    
  
    # load weights
    model_dir = os.path.join(MODEL_DIR + hparams.model_name,dataSetName)
    
    # model.load_weights(f'{model_dir}/13.57_98.h5',by_name='True') 
    model.load_weights(f'{model_dir}/save/best_model18500_0.0554.h5', by_name='True')
        
    # load enroll data
    print("loading data...") 
    (enroll_x, enroll_y) = dataLoad.load_all_data(enroll_dataset, 'enroll')
    
    (eval_x, eval_y) = dataLoad.load_all_data(eval_dataset, 'test')

    # 预测  主要获取说话人嵌入
    enroll_pre = np.squeeze(model.predict(enroll_x))

    eval_pre = np.squeeze(model.predict(eval_x))
    
    # 计算余弦距离
    distances = util.caculate_distance(enroll_dataset,enroll_pre, eval_pre)

    
    if hparams.target=="SI":
    
        # speaker identification
        eval_y_pre = util.speaker_identification(distances, enroll_y)
        
        # compute result
        result = util.compute_result(eval_y_pre, eval_y)
        
        data_dict = {
            'path': eval_dataset[0],
            'predict': eval_y_pre,
            'true': eval_y,
            'isRight':result
        }
        
        data = pd.DataFrame(data_dict)
        data.to_csv('result.csv', index=0)
        score = sum(result)/len(result)
        print(f"score={score}")
    else:
        
        # keras.backend.set_learning_phase(1)
        
        # annotation_file = os.path.join(dataSetDir,"speaker_ver.csv")
        
        annotation_file = './dataset/annonation.csv'
        # util.speaker_verification(model,annotation_file)

        df = pd.read_csv(annotation_file)

        # ismember_true = list(map(int, df['Ismember']))
        
        ismember_true = df['Ismember'].tolist()
        
        # # score_index = distances.argmax(axis=0)
        
        # # 对每列求最大值，得到每句话的最可能说话人
        # y_pred = distances.max(axis=0)
        # # # 每个元素复制nclass次
        # # nclass = len(set(enroll_y))
        # # y_prod = []
        # # for i in range(len(y_pred)):
        # #     y_prod.extend([y_pred[i]]*nclass)
        
        # y_prod = np.array(y_pred)
        # fm, acc, eer = eval_metrics.evaluate(y_prod, ismember_true)
        
        # print(f'eer={eer}\t fm={fm} \t acc={acc}\t')
        
        # np.save('./npys/perfect_noELU.npy',distances)
        ismember_pre,_ = util.speaker_verification(distances, ismember_true)
        
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
    # python run.py --stage="train" --model_name="SEResNet" --target="SV"
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--stage",type=str,required=True,help="train or test",choices=["train","test"])

    parser.add_argument("--model_name",type=str,required=True,help="model's name",choices=["deepSpk","VggVox","SEResNet","AttDCNN"])
    
    parser.add_argument("--target",type=str,required=True,help="SV or SI ,which is used in test stage",choices=["SV","SI"])
    
    # TIMIT
    parser.add_argument("--train_pk_dir",type=str,help="train pickle dir",default="/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/train")
    parser.add_argument("--test_pk_dir",type=str,help="test pickle dir",default="/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/test")  
    
    # parser.add_argument("--train_pk_dir",type=str,help="train pickle dir",default="/home/qmh/Projects/Datasets/LibriSpeech_O/train-clean-100/") # 更换数据集时要修改
    # parser.add_argument("--test_pk_dir",type=str,help="test pickle dir",default="/home/qmh/Projects/Datasets/LibriSpeech_O/test-clean/")     
    
    args = parser.parse_args()
    
    main(args)