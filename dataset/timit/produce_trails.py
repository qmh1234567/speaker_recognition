
import sys
import pandas as pd

fnutt = './eval_utt2spk'  # 评估集所在位置
# ftrial = open('libri_speaker_ver.lst', 'w')
annotation_file = 'speaker_ver.csv'


dictutt = {}

# 测试集
data_dict = {
    'FilePath': [],
    'SpeakerID': [],
    'Ismember': [],
}
            
for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  spk = utt2spk.split(' ')[1]
  if spk not in dictutt:
    dictutt[spk] = spk

print("dictutt=",dictutt)


# 构造SV测试集
for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  utt, spk = utt2spk.split(' ')  # 获取当前句子
  for target in dictutt:   # 遍历所有说话人
    data_dict['FilePath'].append(utt)
    data_dict['SpeakerID'].append(target)
    if target == spk:
      data_dict['Ismember'].append(1)
    else:
      data_dict['Ismember'].append(0)
      

data = pd.DataFrame(data_dict)
data.to_csv(annotation_file,index=0)