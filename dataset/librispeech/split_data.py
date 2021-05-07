
import sys,random
import pandas as pd

dictutt = {}

ftest = './utt2spk'


for line in open(ftest):
  line = line.rstrip('\r\t\n ')
  utt, spk = line.split(' ')
  if spk not in dictutt:
    dictutt[spk] = []
  dictutt[spk].append(utt)


fenroll = open('./enroll_utt2spk', 'w')  # 注册集 每人3句话
feval = open('./eval_utt2spk', 'w')   # 评估集  每人剩下的句子

enroll_dict = {
  'FilePath':[],
  'SpeakerID':[],
}


for key in dictutt:
  utts = dictutt[key]
  random.shuffle(utts)
  for i in range(0, len(utts)):
    line = utts[i] + ' ' + key
    if(i < 3):
      fenroll.write(line + '\n')  # 注册语句是3句
      enroll_dict['FilePath'].append(utts[i])
      enroll_dict['SpeakerID'].append(key)
    else:
      feval.write(line + '\n')

data = pd.DataFrame(enroll_dict)
data.to_csv('./enroll_utt2spk.csv',index=0)

fenroll.close()
feval.close()
