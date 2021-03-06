import keras
import matplotlib.pyplot as plt
import math
import numpy as np
import os
#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
    
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))  # 注意，这里需要和log里面的accuracy一致，不然无法获取到acc
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))
        
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))
    
    def loss_plot(self, loss_type,dataSetName,model_name):
        iters = range(len(self.losses[loss_type]))
        dir1 = os.path.join("./npy/"+dataSetName,model_name)
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        np.save(dir1+"/accuracy.npy",self.accuracy[loss_type])
        np.save(dir1+"/losses.npy",self.losses[loss_type])
        np.save(dir1+"/val_acc.npy",self.val_acc[loss_type])
        np.save(dir1+"/val_loss.npy",self.val_loss[loss_type])
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
           # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        # plt.ylim(0,max(self.losses[loss_type]))
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()