
import sys,random

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

for key in dictutt:
  utts = dictutt[key]
  random.shuffle(utts)
  for i in range(0, len(utts)):
    line = utts[i] + ' ' + key
    if(i < 3):
      fenroll.write(line + '\n')  # 注册语句是3句
    else:
      feval.write(line + '\n')

fenroll.close()
feval.close()
