
import sys

fnutt = './eval_utt2spk'  # 评估集所在位置
ftrial = open('aishell_speaker_ver.lst', 'w')

dictutt = {}
for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  spk = utt2spk.split(' ')[1]
  if spk not in dictutt:
    dictutt[spk] = spk


# 构造SV测试集
for line in open(fnutt):
  utt2spk = line.rstrip('\r\t\n ')
  utt, spk = utt2spk.split(' ')  # 获取当前句子
  for target in dictutt:   # 遍历所有说话人
    if target == spk:
      trial = utt + ' ' + target + ' target'
    else:
      trial = utt + ' ' + target + ' nontarget'
    ftrial.write(trial + '\n')
ftrial.close()
