import matplotlib.pyplot as plt
import numpy as np

filename = './train_acc_eer.txt'

steps = []
eer = []
fm = []
acc = []
moving_eer=[]

def draw_metrics(steps,eer,fm,acc,moving_eer):
    plt.figure()
    plt.plot(steps,eer,label='eer')
    plt.plot(steps,fm,label="f-measure")
    plt.plot(steps,acc,label="accuracy")
    plt.plot(steps,moving_eer,label="moving eer")
    plt.legend(fontsize=12)
    plt.xlabel('steps', fontsize=12)
    plt.ylabel('metrics', fontsize=12)
    plt.ylim([0,1])
    plt.yticks(np.arange(0,1,0.1))
    plt.xticks(np.arange(0, steps[-1],2000))
    plt.show()

def read_movingEER(steps,filename = './train_acc_eer.txt'):
    with open(filename,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(',')
            if(int(line[0]) in steps):
                # print(line[0])
                # 写入数据
                moving_eer.append(round(float(line[1]),2))
    return moving_eer

def read_metrics(filename='./test_log.txt'):
    with open(filename,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(',')
            # 写入数据
            steps.append(int(line[0]))
            eer.append(round(float(line[1]),2))
            fm.append(round(float(line[2]),2))
            acc.append(round(float(line[3]),2))
    return steps,eer,fm,acc


if __name__ == "__main__":
    steps,eer,fm,acc = read_metrics()
    moving_eer = read_movingEER(steps)
    draw_metrics(steps,eer,fm,acc,moving_eer)
            
        # print(len(steps))
        # draw_metrics(steps,eer,fm,acc)
        # # steps.append(line[0])
        # # eer.append()