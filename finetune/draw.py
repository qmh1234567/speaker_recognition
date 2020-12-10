import matplotlib.pyplot as plt
import numpy as np

filename = './../checkpoint/deepSpk/save/train_acc_eer.txt'

steps = []
eer = []
fm = []
acc = []
mv_eer=[]

def draw_metrics(steps,eer,fm,acc):
    plt.figure()
    plt.plot(steps,eer,label='eer')
    plt.plot(steps,fm,label="f-measure")
    plt.plot(steps,acc,label="accuracy")
    # plt.plot(steps,mv_eer,label="moving eer")
    plt.legend(fontsize=12)
    plt.xlabel('steps', fontsize=12)
    plt.ylabel('metrics', fontsize=12)
    plt.ylim([0,1])
    plt.yticks(np.arange(0,1,0.1))
    # plt.xticks(np.arange(0, steps[-1],2000))
    plt.show()
    plt.savefig("first1.png")

def read_movingEER(steps,filename =filename):
    with open(filename,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(',')
            if(int(line[0]) in steps):
                # print(line[0])
                # 写入数据
                mv_eer.append(round(float(line[1]),2))
    return mv_eer

def read_metrics(filename=filename):
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


def plot_acc(file=filename):
    step = []
    eer = []
    fm = []
    acc = []
    mov_eer=[]
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           eer.append(float(line.split(",")[1]))
           fm.append(float(line.split(",")[2]))
           acc.append(float(line.split(",")[3]))
           if mv == 0:
               mv = float(line.split(",")[1])
           else:
               mv = 0.1*float(line.split(",")[1]) + 0.9*mov_eer[-1]
           mov_eer.append(mv)
    p1, = plt.plot(step, fm, color='black',label='F-measure')
    p2, = plt.plot(step, eer, color='blue', label='EER')
    p3, = plt.plot(step, acc, color='red', label='Accuracy')
    p4, = plt.plot(step, mov_eer, color='yellow', label='Moving_Average_EER')
    plt.xlabel("Steps")
    plt.ylabel("metrics")
    plt.legend(handles=[p1,p2,p3,p4],labels=['F-measure','EER','Accuracy','moving_eer'],loc='best')
    plt.show()
    plt.savefig("second.png")
    
    
    
if __name__ == "__main__":
    # steps,eer,fm,acc = read_metrics()
    # draw_metrics(steps,eer,fm,acc)
    plot_acc()    
        # print(len(steps))
        # draw_metrics(steps,eer,fm,acc)
        # # steps.append(line[0])
        # # eer.append()