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
from random_batch import stochastic_mini_batch,data_catalog,split_dataset
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
LEARN_RATE = 0.0001 # 学习率 



TEST_PER_EPOCHS = 200   # 多少轮测试一下
SAVE_PER_EPOCHS = 2000   # 多少轮保存一下

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
        model = Att_DCNN.Att_DCNN().proposed_model(input_shape)
    # print(model.summary())
    return model


def train(model,dataLoad,util,hparams):
    Num_Iter = 100001
    current_iter = 0
    grad_steps = 0
    lasteer = 10
    
    dataSetName = hparams.train_pk_dir.split("/")[-3].split("_")[0].lower()
    
    model_dir = os.path.join(MODEL_DIR + hparams.model_name,dataSetName)
    
    best_model_dir = os.path.join(model_dir,"save")
    if not os.path.exists(best_model_dir):
            os.mkdir(best_model_dir)
    
    
    # 没有验证集时，需要将注册集切分一部分出来作为验证集
    # train_paths,val_paths = split_dataset(hparams.train_pk_dir)
    
    # train_dataset = data_catalog(train_paths)
    
    # val_dataset = data_catalog(val_paths)
    
    train_dataset = data_catalog(hparams.train_pk_dir)
    test_dataset = data_catalog(hparams.test_pk_dir)
    
    # 配置模型
    model.load_weights(f'{model_dir}/6.55_300.h5', by_name='True') # 加载预训练权重
    
    sgd = optimizers.SGD(lr=LEARN_RATE,momentum=0.9) #TIMIT libri-seresnet
    model.compile(optimizer=sgd, loss=deep_speaker_loss)
     
    
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
            
            fm1, acc1, eer1 = eval_model(model,test_dataset, hparams.batch_size*hparams.triplet_per_batch, check_partial=True)
            logging.info('test training data EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer1, fm1, acc1))
            
            # with open(f'{best_model_dir}/train_acc_eer.txt', "a") as f:
            #     f.write("{0},{1},{2},{3}\n".format(grad_steps, eer1, fm1, acc1))

        # 每200步，测试一下模型
        if (grad_steps ) % TEST_PER_EPOCHS == 0 :
            
            fm, acc, eer = eval_model(model,test_dataset,hparams.batch_size*hparams.triplet_per_batch,check_partial=False) 
            
            
            logging.info('== Testing model after batch #{0}'.format(grad_steps))
            logging.info('EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer, fm, acc))
            
            # with open(f'{best_model_dir}/test_log.txt', "a") as f:
            #     f.write("{0},{1},{2},{3}\n".format(grad_steps, eer, fm, acc))

        # 每1000步保存一下模型
        if (grad_steps ) % SAVE_PER_EPOCHS == 0:
            
            model_path = '{0}/model_{1}_{2:.4f}.h5'.format(model_dir, grad_steps, eer)
            
            model.save_weights(model_path)
            
            # 测试模型
            final_eer= getEER(model,dataLoad,hparams,util,model_path)
            logging.info('final_EER = {0:.4f} ref_EER = {1:.4f} '.format(final_eer,eer))
            
            with open(f'{best_model_dir}/eval_log.txt', "a") as f:
                f.write("{0},{1:.4f},{2:.4f}\n".format(grad_steps, final_eer, eer))
            
            if final_eer < lasteer:
                files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                      map(lambda f: os.path.join(best_model_dir, f), os.listdir(best_model_dir))),
                               key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                lasteer = final_eer  # 更新eer
                
                for file in files[:-4]:
                    logging.info("removing old model: {}".format(file))
                    os.remove(file)
                    
                model.save_weights(best_model_dir+'/best_model{0}_{1:.4f}.h5'.format(grad_steps, final_eer))
                
        grad_steps += 1


def main(hparams):    
        
    model = createModel(hparams.model_name)
    
    annotation_file = '../dataset/annonation.csv'

    dataLoad = DataLoad.DataLoad(annotation_file)
    
    util = Util.Util()
    
    if hparams.stage.startswith('train'):
        
        train(model,dataLoad,util,hparams)



def getEER(model,dataLoad,hparams,util,model_path):
    
    eval_dataset, enroll_dataset = dataLoad.createTestDataSet(
            hparams.test_pk_dir,target=hparams.target, split_ratio=0.5)
    
    model.load_weights(model_path, by_name='True')
        
    # load enroll data
    print("loading data...") 
    (enroll_x, enroll_y) = dataLoad.load_all_data(enroll_dataset, 'enroll')
    
    (eval_x, eval_y) = dataLoad.load_all_data(eval_dataset, 'test')

    # 预测  主要获取说话人嵌入
    enroll_pre = np.squeeze(model.predict(enroll_x))

    eval_pre = np.squeeze(model.predict(eval_x))
    
    # 计算余弦距离
    distances = util.caculate_distance(enroll_dataset,enroll_pre, eval_pre)

    df = pd.read_csv(dataLoad.ANNONATION_FILE)

    
    ismember_true = df['Ismember'].tolist()
    
    ismember_pre,eer = util.speaker_verification(distances, ismember_true)
    
    return eer
        



if __name__ == "__main__":
    
    # nohup python run.py --stage="train" --model_name="deepSpk" --target="SI" > output.out &
    # python run.py --stage="test" --model_name="deepSpk" --target="SV"
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--stage",type=str,required=True,help="train or test",choices=["train","test"])

    parser.add_argument("--model_name",type=str,required=True,help="model's name",choices=["deepSpk","VggVox","SEResNet","AttDCNN"])
    
    parser.add_argument("--target",type=str,required=True,help="SV or SI ,which is used in test stage",choices=["SV","SI"])
    
    # parser.add_argument("--train_pk_dir",type=str,help="train pickle dir",default="/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/train/") # 更换数据集时要修改
    
    # parser.add_argument("--test_pk_dir",type=str,help="test pickle dir",default="/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/test")  # 更换数据集时要修改
    
    # librispeech
    parser.add_argument("--train_pk_dir",type=str,help="train pickle dir",default="/home/qmh/Projects/Datasets/LibriSpeech_O/train-clean-100/") # 更换数据集时要修改
    
    parser.add_argument("--test_pk_dir",type=str,help="test pickle dir",default="/home/qmh/Projects/Datasets/LibriSpeech_O/test-clean/")  
    
    parser.add_argument("--batch_size",type=int,default=16)
    
    parser.add_argument("--triplet_per_batch",type=int,default=3)
    args = parser.parse_args()
     
    main(args)