from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import tqdm
import pandas as pd


class Util():
    
    # 计算余弦距离
    def caculate_distance(self,enroll_dataset, enroll_pre, test_pre):
        # print("enroll_pre.shape=", enroll_pre.shape)
        dict_count = Counter(enroll_dataset[1])

        # each person get a enroll_pre
        speakers_pre = []
        # 得到注册说话人 remove repeat
        enroll_speakers = list(set(enroll_dataset[1]))

        enroll_speakers.sort(key=enroll_dataset[1].index)
      
        for speaker in enroll_speakers:
            start = enroll_dataset[1].index(speaker)
            speaker_pre = enroll_pre[start:dict_count[speaker]+start]
            speakers_pre.append(np.mean(speaker_pre, axis=0))

        enroll_pre = np.array(speakers_pre)
        # print("new_enroll_pre.shape=", enroll_pre.shape)  #(168,512)  168人 512维
        # caculate distance
        distances = []
        # print("test_pre.shape=", test_pre.shape)  #(840,512) 840句话  512维
        for i in range(enroll_pre.shape[0]):
            temp = []
            for j in range(test_pre.shape[0]):
                x = enroll_pre[i]
                y = test_pre[j]
                s = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))  # 计算余弦距离
                temp.append(s)
            distances.append(temp)
        distances = np.array(distances)
        # print("distances.shape=", distances.shape) #(168,840)  168人  840句 
        return distances
    
    
    # 计算结果
    def compute_result(self,y_pre, y_true):
        result = []
        for index, x in enumerate(y_pre):
            result.append(1 if x == y_true[index] else 0)
        return result
    
    # 得到评估指标
    def evaluate_metrics(self,y_true, y_pre):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        # np.save('./npy/timit/deepSpk/dp_fpr_noBN.npy',fpr)
        # np.save('./npy/timit/deepSpk/dp_tpr_noBN.npy',tpr)
        # plt.figure()
        # plt.plot(fpr, tpr, color='green', label='ROC')
        # plt.plot(np.arange(1, 0, -0.01), np.arange(0, 1, 0.01))
        # plt.legend()
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.xlabel('fpr')
        # plt.ylabel('tpr')
        # plt.title(f'ROC curve, AUC score={auc}')
        # plt.show()

        threshold_index = np.argmin(abs(1-tpr - fpr))
        threshold = thresholds[threshold_index]
        eer = ((1-tpr)[threshold_index]+fpr[threshold_index])/2
        auc_score = metrics.roc_auc_score(y_true, y_pre, average='macro')

        y_pro = [1 if x > threshold else 0 for x in y_pre]
        acc = metrics.accuracy_score(y_true, y_pro)
        prauc = metrics.average_precision_score(y_true, y_pro, average='macro')
        return y_pro, eer, prauc, acc, auc_score

    # 测试SV
    def speaker_verification(self,distances, ismember_true):
        
        score_index = distances.argmax(axis=0)
        
        # 对每列求最大值，得到每句话的最可能说话人
        distance_max = distances.max(axis=0)
        # 由于余弦距离可能是负值，故需要平移一下
        distance_max = (distance_max + 1) / 2
        
        np.save("./dataset/SE_y_pre.npy",distance_max)
        np.save("./dataset/SE_y_true.npy",ismember_true)
        
        # np.save("./dataset/Att_y_pre.npy",distance_max)
        # np.save("./dataset/Att_y_true.npy",ismember_true)
        
        # np.save("./dataset/Deep_y_pre.npy",distance_max)
        # np.save("./dataset/Deep_y_true.npy",ismember_true)
     
        y_pro, eer, prauc, acc, auc_score = self.evaluate_metrics(
            ismember_true, distance_max)
        
        print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')
        return y_pro,eer
    
    # 测试SI 
    def speaker_identification(self, distances, enroll_y):
        #  remove repeat
        new_enroll_y = list(set(enroll_y))
        new_enroll_y.sort(key=list(enroll_y).index)
        # print("new_enroll_y=",new_enroll_y)
        #  return the index of max distance of each sentence
        socre_index = distances.argmax(axis=0)
        y_pre = []
        for i in socre_index:
            y_pre.append(new_enroll_y[i])
        # print("y_pre=",y_pre)
        return y_pre  # 获得每一句的说话人
    
    
    # map utt to embedding
    def map_utt2embedding(self,utt,label,model):
        with open(utt,"rb") as f:
            vec = load_dict["LogMel_Features"]
            vec = vec[:,:,np.newaxis]
            pre = np.squeeze(model.predict(vec))
        return pre
    
    
    # def speaker_verification(self,model,annotation_file):
        
    #     df = pd.read_csv(annotation_file)

    #     y_true = list(map(int, df['Istarget']))
        
    #     paths = df['FilePath'].tolist()
        
    #     labels = df['SpeakerID'].tolist()
        
    #     # 列表去重
        
    #     # for index in tqdm.tqdm(range(0,len(y_true))):
            
    #     print(labels[:20])
        
    #     exit()
    #     print(len(ismember_true))
