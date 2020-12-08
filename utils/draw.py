import pandas as pd
import numpy as np
import constants as c
from sklearn import metrics
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
    distance = np.load('./npys/perfect.npy')
    distance_nodrp = np.load('./npys/perfect_nodrop.npy')
    distance_nose = np.load('./npys/perfect_nose.npy')
    distance_nol2 = np.load('./npys/perfect_nol2.npy')
    distance_noELU = np.load('./npys/perfect_noELU.npy')
    distances = (distance,distance_nodrp,distance_nose,distance_nol2,distance_noELU)
    df = pd.read_csv(c.ANNONATION_FILE)
    ismember_true = list(map(int,df["Ismember"]))
    ismember_pre = speaker_verification(distances, ismember_true)