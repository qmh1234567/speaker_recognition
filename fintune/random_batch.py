# 随机选择batch,从数据集中随机选一个当作anchor,再从同一个speaker的语音中随机选一个当作positive
# 再从不同speaker的语音中随机选一个当作negative

import numpy as np
import pandas as pd
import glob
import os
import pickle

NUM_FRAMES = 300
BATCH_SIZE = 32


def data_catalog(dataset_dir, pattern='*.pickle'):
    dataSet = pd.DataFrame()

    audio_paths = [pickle for pickle in glob.iglob(dataset_dir +"/*.pickle")]

    dataSet['filename'] = [pickle for pickle in audio_paths]  # normalize windows paths

    dataSet['speaker_id'] = [os.path.basename(pickle).split("_")[0] for pickle in audio_paths]

    num_speakers = len(dataSet['speaker_id'].unique())

    # print('Found {} files with {} different speakers.'.format(str(len(dataSet)).zfill(7), str(num_speakers).zfill(5)))
    # print(libri.head(10))

    # dataSet.to_csv("test.csv", index=0)
    
    return dataSet

class MiniBatch:
    def __init__(self,dataset,batch_size,unique_speakers=None):
        if unique_speakers is None:
            unique_speakers = list(dataset['speaker_id'].unique())
        num_triplets = batch_size

        anchor_batch = None
        positive_batch = None
        negative_batch = None

        for i in range(num_triplets):
            two_different_speakers = np.random.choice(unique_speakers,size=2,replace=False)
            # 选一个speaker作为anchor
            anchor_positive_speaker = two_different_speakers[0]
            # 另外一个speaker作为negative
            negative_speaker = two_different_speakers[1]
            # 读取音频 从anchor的音频中随机抽取两个
            anchor_positive_file = dataset[dataset['speaker_id']==anchor_positive_speaker].sample(n=2,replace=False)
            
            # 第一个音频作为anchor,第二个音频作为positive
            anchor_df = pd.DataFrame(anchor_positive_file[0:1])  
            anchor_df['training_type'] = 'anchor'
            positive_df = pd.DataFrame(anchor_positive_file[1:2])
            positive_df['training_type'] = 'positive'

            # 从negative的音频中抽取一个
            negative_df = dataset[dataset['speaker_id'] == negative_speaker].sample(n=1)
            negative_df['training_type'] = 'negative'

            ## 构建训练集的表格
            if anchor_batch is None:
                anchor_batch = anchor_df.copy() 
            else:
                anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)  # 添加到表格中去

            if positive_batch is None:
                positive_batch = positive_df.copy()
            else:
                positive_batch = pd.concat([positive_batch, positive_df], axis=0)

            if negative_batch is None:
                negative_batch = negative_df.copy()
            else:
                negative_batch = pd.concat([negative_batch, negative_df], axis=0)

        # 做成一个大的训练集表 32*3=96行
        self.dataset_batch = pd.DataFrame(pd.concat([anchor_batch,positive_batch,negative_batch],axis=0))
        # self.dataset_batch.to_csv("test1.csv",index=0)
        self.num_triplets = num_triplets
    
    # X, Y = [], []
    # bar = Bar('loading data', max=len(labels),fill='#', suffix='%(percent)d%%')
    # for index, pk in enumerate(path):
    #     bar.next()
    #     try:
    #         with open(pk, "rb") as f:
    #             load_dict = pickle.load(f)
    #             x = load_dict["LogMel_Features"]
    #             x = x[:, :, np.newaxis]
    #             X.append(x)
    #             Y.append(labels_to_id[labels[index]])
    #     except Exception as e:
    #         print(e)
    # X = np.array(X)
    # Y = np.eye(num_class)[Y]
    # bar.finish()

    def to_inputs(self):
        new_x = []

        for i in range(len(self.dataset_batch)):
            filename = self.dataset_batch[i:i + 1]['filename'].values[0]
            with open(filename,"rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                new_x.append(x)

        x = np.array(new_x) #(batchsize, num_frames, 64, 1) （96，299，40，1）
        y = self.dataset_batch['speaker_id'].values  #（96）
    
        # anchor examples [speakers] == positive examples [speakers]
        np.testing.assert_array_equal(y[0:self.num_triplets], y[self.num_triplets:2 * self.num_triplets])

        return x, y


def stochastic_mini_batch(dataset, batch_size=BATCH_SIZE,unique_speakers=None):
    mini_batch = MiniBatch(dataset, batch_size,unique_speakers)
    return mini_batch

def main():
    dataset_dir = "/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/train"
    dataset = data_catalog(dataset_dir)

    batch = stochastic_mini_batch(dataset,BATCH_SIZE)
    batch_size = BATCH_SIZE*3
    x,y = batch.to_inputs()
    b = x[0]
    input_shape = (b.shape[0],b.shape[1],b.shape[2])


if __name__ == "__main__":
    main()