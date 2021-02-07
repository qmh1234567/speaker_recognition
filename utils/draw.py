import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
import os

def evaluate_metrics(y_true, y_pres):
    plt.figure()
    aucs= [] 
    for y_pre in y_pres:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr,label='AUC')
        aucs.append(round(auc,4))

    # plt.plot(np.arange(1, 0, -0.01), np.arange(0, 1, 0.01))
    plt.legend([f'seresnet,auc={aucs[0]}',f'seresnet-nodrop,auc={aucs[1]}',
    f'seresnet-nose,auc={aucs[2]}',f'seresnet-nol2,auc={aucs[3]}',f'seresnet-noELU,auc={aucs[4]}'])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('false positive rate (fpr)')
    plt.ylabel('true positive rate (tpr)')
    plt.title(f'ROC-AUC curve')
    plt.show()

    threshold_index = np.argmin(abs(1-tpr - fpr))
    threshold = thresholds[threshold_index]
    eer = ((1-tpr)[threshold_index]+fpr[threshold_index])/2
    print(eer)
    auc_score = metrics.roc_auc_score(y_true, y_pre, average='macro')

    y_pro = [1 if x > threshold else 0 for x in y_pre]
    acc = metrics.accuracy_score(y_true, y_pro)
    prauc = metrics.average_precision_score(y_true, y_pro, average='macro')
    return y_pro, eer, prauc, acc, auc_score


def loss_plot(dataSetName,model_name):
    dir1 = os.path.join("../npy/"+dataSetName,model_name)
    accuracy = np.load(dir1+"/accuracy.npy")
    losses = np.load(dir1+"/losses.npy")
    val_acc = np.load(dir1+"/val_acc.npy")
    val_loss = np.load(dir1+"/val_loss.npy")
    iters = range(len(losses))
    plt.figure()
    # acc
    plt.plot(iters, accuracy, label='train acc')
    # loss
    plt.plot(iters, losses, label='train loss')
    # val_acc
    plt.plot(iters,val_acc, label='val acc')
    # val_loss
    plt.plot(iters, val_loss, label='val loss')
    # plt.ylim(0,max(self.losses[loss_type]))
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.show()
    
        

def main():
    dataSetName = "librispeech"
    model_names1 = ["AttDCNN","SEResNet","deepSpk"]
    model_names = ["Attentive DCNN","SECNN","Deep Speaker"]
    plt.figure()
    for i in range(len(model_names)):
        dir1 = os.path.join("../npy/"+dataSetName,model_names1[i])
        accuracy = np.load(dir1+"/accuracy.npy")
        losses = np.load(dir1+"/losses.npy")
        val_acc = np.load(dir1+"/val_acc.npy")
        val_loss = np.load(dir1+"/val_loss.npy")
        iters = range(len(losses))
        # acc
        # plt.plot(iters, accuracy, label=model_name+'_train acc')
        # loss
        # plt.plot(iters, losses, label=model_name+'_train loss')
        # val_acc
        plt.plot(iters,val_acc, label=model_names[i]+'_val acc')
        # val_loss
        plt.plot(iters, val_loss, label=model_names[i]+'_val loss')
    # plt.ylim(0,max(self.losses[loss_type]))
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    
    # dataSetName = sys.argv[1]
    # model_name = sys.argv[2]
    
    # loss_plot(dataSetName,model_name)
    main()
    # python run.py --stage="test" --model_name="deepSpk" --target="SV"