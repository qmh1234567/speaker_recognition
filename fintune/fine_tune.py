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
from test_model import eval_model

import sys
sys.path.append("../") 
import usedModels.Deep_Speaker as Deep_Speaker
import usedModels.VggVox as VggVox
import usedModels.SE_ResNet as SE_ResNet
import usedModels.Att_DCNN as Att_DCNN

import utils.DataLoad as DataLoad
import utils.Util as Util
import utils.LossHistory as LossHistory
from random_batch import stochastic_mini_batch,data_catalog
from triplet_loss import deep_speaker_loss
import logging
import pickle
from time import time

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)

MODEL_DIR = './../checkpoint/' # 模型保存目录
LEARN_RATE = 0.001 # 学习率



TEST_PER_EPOCHS = 200   # 多少轮测试一下
SAVE_PER_EPOCHS = 200   # 多少轮保存一下

# 设置log输出到控制台
logger = logging.getLogger()   
logger.setLevel(logging.INFO)  

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


def train(model,dataLoad,hparams):
    Num_Iter = 1000000
    current_iter = 0
    grad_steps = 0
    lasteer = 10
    model_dir = MODEL_DIR + hparams.model_name
    
    best_model_dir = os.path.join(model_dir,"save")
    if not os.path.exists(best_model_dir):
            os.mkdir(best_model_dir)
            
    train_dataset = data_catalog(hparams.train_pk_dir)
    test_dataset = data_catalog(hparams.test_pk_dir)
    
    # 配置模型
    model.load_weights(f'{model_dir}/best.h5', by_name='True') # 加载预训练权重
    
    model.compile(optimizer='adam', loss=deep_speaker_loss)
     
    while current_iter <Num_Iter:
        current_iter += 1
        
        batch= stochastic_mini_batch(train_dataset,hparams.batch_size)
        x,y = batch.to_inputs()  #(96,299,40,1)
        # y = np.random.uniform(size=(x.shape[0],1))  # (96)
    
        logging.info('== Presenting step #{0}'.format(grad_steps))
        orig_time = time()
        
        loss = model.train_on_batch(x, y)
        
        logging.info('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, loss))
        # 每10步评估一下模型
        if (grad_steps) % 10 == 0:
            
            fm1, tpr1, acc1, eer1 = eval_model(model,train_dataset, hparams.batch_size*hparams.triplet_per_batch, check_partial=True)
            logging.info('test training data EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer1, fm1, acc1))
            
            with open('./train_acc_eer.txt', "w") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer1, fm1, acc1))
                
        # 每200步，测试一下模型
        if (grad_steps ) % TEST_PER_EPOCHS == 0 :
            
            fm, tpr, acc, eer = eval_model(model,test_dataset,hparams.batch_size*hparams.triplet_per_batch,check_partial=False)
            
            logging.info('== Testing model after batch #{0}'.format(grad_steps))
            logging.info('EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer, fm, acc))
            
            with open('./test_log.txt', "w") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer, fm, acc))

        # 每200步保存一下模型
        if (grad_steps ) % SAVE_PER_EPOCHS == 0:
            
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(model_dir, grad_steps, loss))
            
            if eer < lasteer:
                files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                      map(lambda f: os.path.join(best_model_dir, f), os.listdir(best_model_dir))),
                               key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                lasteer = eer  # 更新eer
                
                for file in files[:-4]:
                    logging.info("removing old model: {}".format(file))
                    os.remove(file)
                    
                model.save_weights(best_model_dir+'/best_model{0}_{1:.5f}.h5'.format(grad_steps, eer))
                
        grad_steps += 1


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
    
    parser.add_argument("--batch_size",type=int,default=32)
    
    parser.add_argument("--triplet_per_batch",type=int,default=3)
    args = parser.parse_args()
    
    main(args)