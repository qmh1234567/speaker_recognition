# Spaker_recognition

#### 运行环境说明：
  - 系统：ubuntu
  - IDE: vscode
  - gpu: GeForce RTX 2070

#### TIMIT数据集上的声纹识别实验

> TIMIT数据是收费的，但是国外有个大学网站提供了免费下载地:
[TIMIT数据下载地址](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3/tech&hit=1&filelist=1)
文件大小440M，是完整的数据库。
- 由于默认的目录名字和文件后缀都是大写的，而本项目只支持小写，可参考[shell递归遍历目录，修改文件名或者目录名](https://blog.csdn.net/qq_28228605/article/details/109963278)，将它们改为小写。

#### 1. 安装所需依赖包

```bash
$ pip install -r requirements.txt
```

#### 2. 项目目录说明：
  |  目录名   | 说明  |
|  ----  | ----  |
| dataset  | 暂无 |
| fintune  | 与微调相关的文件夹 |
| imgs | 实验中生成的一些图片|
| usedModels | 存储各种模型 |
| utils | 工具包 |
| run.py | 预训练与测试脚本 |


#### 3. 运行utils目录下的预处理文件`preprocess.py`
 - 运行命令：
   - `python preprocess.py --in_dir=TIMIT/train/ --pk_dir=/TIMIT_OUTPUT/train/ --data_type=mit`
  - 参数说明：
      - --in_dir: TIMIT数据集的目录，需要指定到train和test
      - --pk_dir: 预处理后的pickle文件保存目录，也是需要指定到train和test
      - --data_type:数据集类型，选择`mit`
  - 遇到的问题1：
    ```bash
    /home/qmh/Projects/anaconda3/envs/normal/lib/python3.8/site-packages/pydub/utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
    warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)
    ```
    - 表示没有找到ffmpeg,此时需要使用`sudo apt install ffmpeg`命令安装`ffmpeg`
  - 遇到的问题2：
    ```bash
        audio, sample_rate = vad_ex.read_libri(path)
    File "/home/qmh/Projects/ResCNN_speaker_recognition/vad_ex.py", line 25, in read_libri
      mf = AudioSegment.from_file(path, "wav")
    File "/home/qmh/Projects/anaconda3/envs/normal/lib/python3.8/site-packages/pydub/audio_segment.py", line 723, in from_file
      raise CouldntDecodeError(
    pydub.exceptions.CouldntDecodeError: Decoding failed. ffmpeg returned error code: 1
    ```
    - 表示音频文件格式编码错误,由于TIMIT数据集中的音频文件后缀是WAV，需要转换编码为wav.在数据集的train目录下编写`run.sh`，具体内容如下：
    ```bash
    #!/bin/bash
      # 转成wav
      function convertTowav(){
      # echo ffmpeg -loglevel panic -y -i $1 $1
      name=$(basename "$1" .wav)
      new=$name"_1.wav"
      #echo ffmpeg -loglevel panic -y -i $1 $new
      ffmpeg -loglevel panic -y -i $1 $new
      mv $new $1
      }

      # 遍历文件夹
      function travFolder(){
      flist=`ls $1`   # 第一级目录
      cd $1        
      for f in $flist  # 进入第一级目录
      do
          if test -d $f  # 判断是否还是目录
          then 
          travFolder $f # 是则继续递归
          else
          if [ "${f##*.}"x = "wav"x ] 
          then
                  convertTowav $f # 否则进行转化
          fi
          fi
      done
      cd ../     # 返回目录
      }
      dir='./'
      travFolder $dir
    ```
- 预处理逻辑：
  ```python
   # 对每个音频进行预处理
        for path in path_list:
            # 去静音
            wav_arr, sample_rate = self.vad_process(path)
            if sample_rate != 16000:
                print("sample rate do meet the requirement")
                exit()
            # padding 音频裁减
            wav_arr = self.cut_audio(wav_arr,sample_rate)   
            # 提取特征并保存
            self.create_pickle(path, wav_arr, sample_rate)
  ```

- 去静音效果对比
  - 去静音前音频：
  ![1](/imgs/before.png)
  - 去静音后音频
  ![1](/imgs/after_VAD.png)
  - 填充到3s后音频
  ![1](/imgs/after_Padding.png)
- 语谱图
  ![1](/imgs/spectrum.png)

#### 4. 运行预训练脚本`run.py`
  - 运行命令：`python run.py --stage="test" --model_name="deepSpk" --target="SV"` 
  - 参数说明：
    - --stage 表示实验阶段，train是训练阶段，test是测试阶段
    - --model_name 表示模型名字，本项目的所有模型都在usedModels文件夹下。您也可以根据自己的需要设计模型。
    - --target 表示实验目标，SV是说话人确认，SI是说话人辨认。该参数仅在test阶段起作用。


#### 5. usedModels目录下的模型介绍
- DeepSpeaker模型：[论文地址](https://arxiv.org/abs/1705.02304)
- Rest34 和 Rest50模型： [论文地址](https://arxiv.org/abs/1806.05622)
  
#### 6.各个模型的实验结果
##### Deep Speaker模型实验结果
- loss曲线和acc曲线：
![1](/imgs/dpk_acc_loss.png)
- SI实验：
score=0.875
- SV实验：   
  - eer=0.08155405405405403	 
  - prauc=0.3926588193540755 	 
  - acc=0.9164556962025316	 
  - auc_score=0.9761081081081081	
  - score=0.9164556962025316


##### VggVox模型实验结果
- 待添加


  
##### 提出的SE-ResNet模型实验结果
- loss曲线和acc曲线：
![1](/imgs/seResNet_acc_loss.png)
- SI实验：
score=0.945
- SV实验：   
  - eer=0.04695945945945948	 
  - prauc=0.570774619855107 	 
  - acc=0.9563291139240506	 
  - auc_score=0.992331081081081	
  - score=0.9563291139240506

#### 使用Triplet loss对上述模型进行微调

 > 主要参考这位大佬的[deep speaker 实验](https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system)

 ##### 自己的改动
 - 预处理和预训练部门均使用的自己的代码。微调部门参考了大佬的代码。
 - 微调目录内容介绍：
    |  脚本名   | 说明  | 改动说明 |
    |  ----  | ----  |---|
    | draw.py  | 绘制各种指标的曲线 | 一致 |
    | eval_metrics.py  | 计算各种指标的函数| 没改,尝试改过，但是改过后计算出来的指标有问题，后面解决 |
    | fine_tune.py | 微调的主脚本 | 改动了训练的代码，小改 |
    | random_batch.py | 生成批次的脚本 | 大改，修改了数据加载方式 |
    | test_model.py | 评估模型的脚本 | 小改，修改了余弦函数的计算|
    | triplet_loss.py | 三元组损失计算脚本 | 小改，修改了余弦函数的计算


 ##### 自己的踩坑记录
 1. 训练结果中acc非常低,而且波动幅度过大，失败的训练结果如图：
 ![1](/imgs/failedResult.png)
    - 第一反映是自己的评估模型函数有问题，经过一上午的反复检查和理解代码，终于排查出了问题。原因在于此函数：
  
      ```python
      # 构建输入和输出
      def to_inputs(dataset_batch,num_triplets):
            new_x = []

            for i in range(len(dataset_batch)):
                filename = dataset_batch[i:i + 1]['filename'].values[0]
                with open(filename,"rb") as f:
                    load_dict = pickle.load(f)
                    x = load_dict["LogMel_Features"]
                    x = x[:, :, np.newaxis]
                    new_x.append(x)

            x = np.array(new_x) #（1530，299，40，1）
            # y = dataset_batch['speaker_id'].values  #（1530）
            new_y = np.hstack(([1],np.zeros(num_neg)))  # 1 positive, num_neg negative 这里需要注意，没有anchor的参与！！！
            y = np.tile(new_y, num_triplets)  # (one hot) （1500）  
            return x, y
      ```
     - 当时在选择随即批次的那个脚本中也有相似函数，故我的第一直觉是x和y的shape要一样，于是自作聪明的写了`new_y = np.hstack(([1,1],np.zeros(num_neg)))`但实际上大佬的代码注释已经很清楚了，这里的label不需要考虑anchor了。因为在计算相似度时是用同一个anchor嵌入和所有的postive与negative的嵌入计算的余弦相似度。而这个相似度数组的长度是len(postive+negative)，即y_pred.
     - 修改后得到的结果如下图：
     ![1](/imgs/successedResult.png)

2. 使用其他模型，如SEResNet时，出现ACC非常低的情况，除了acc其他指标也不正常。

   - 这个错误我也排查了一会，首先数据集是不会有问题的，对于所有模型都是一样的。其次模型的加载也没有问题。唯一的问题也在评估模型的函数。我看了一下基本都是通用的，没有针对具体模型的函数。同时我也打印了一下y_pred,发现其他模型的y_pred非常大，数值为好几百,而deepSpk的y_pred在0到1之间。故猜想可能是余弦距离的计算函数出了问题，于是换成了常用的余弦距离计算函数，然后结果就恢复正常了。
    
    ```python
      # 计算余弦距离 
    def batch_cosine_similarity(x1,x2):
        # https://en.wikipedia.org/wiki/Cosine_similarity
        # 1 = equal direction ; -1 = opposite direction
        # 方法1 方法1的结果会超过1
        # mul = np.multiply(x1, x2)
        # s = np.sum(mul,axis=1)
        # 方法2 
        s1 = []
        for i in range(0,x1.shape[0]):
            sm = np.dot(x1[i], x2[i])/(np.linalg.norm(x1[i])*np.linalg.norm(x2[i]))# 计算余弦距离
            s1.append(sm)  
        return np.array(s1)
    ``` 
  3.发现其他模型使用triplet loss微调后，效果没有改进。如这是我自己定义的AttDCNN模型使用微调，跑了8万多步，获得的一个最好模型，但该模型在测试集上eer的值不降反升。从原来的7.01%变成了10.01%.

  - 这个问题的原因还在排查中。
     
     - 首先检查了一下triple_loss脚本，发现里面的计算余弦距离的函数，我不太理解。于是换成了自己能够理解的函数。 改动之后发现训练loss显著降低，从原来的十几、二十几降低到1以下。
        ```python
        # # 计算余弦距离 
        def batch_cosine_similarity(x1,x2):
            dot1 = K.batch_dot(x1, x2, axes=1)  # a*b
            dot2 = K.batch_dot(x1, x1, axes=1) # a*a
            dot3 = K.batch_dot(x2, x2, axes=1) # b*b
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon()) # sqrt(a*a * b*b) K.epsilon() 是为了防止为0
            return dot1 / max_  #  a*b/sqrt(a*a * b*b)
        ```
     - 其次检查了梯度下降方法和学习率，发现用的是adam,学习率默认是0.01，猜测可能是梯度下降策略不对或者学习率太大，于是改成了梯度下降，学习率初始值是0.001。 修改后发现成功降低了eer，由原来的7.0%降低到6.9%。