# ResCNN_speaker_recognition
运行准备：
1. 下载library_speech数据集，地址：http://www.openslr.org/resources/12/。
共三个文件夹：	train-clean-100.tar.gz, test-clean.tar.gz,	dev-clean.tar.gz。
2. 将convert_flac_2_wav_sox.sh复制到数据集所在目录，该文件是将.flac转化为.wav. 参考地址:https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system
  
3. 在train-clean和test-clean目录下新建wav文件夹，将所有音频文件都剪切进去。

运行说明:
1. 修改constants.py 中的数据集路径:TRAIN_DEV_SET_LB和TEST_SET_LB,改成您下载的library speech的相应目录.
2. 运行preprocess.py,特征文件生成在下载的library speech的对应目录下的npy文件夹.
3. 运行python run.py train_lb命令进行训练模型,模型有两个:rescnn和deepspeaker,目前两个的SI效果都不太好.
4. 运行python run.py test_lb命令进行测试,设置constants.py中的TARGET切换SI或SV.

#### 运行环境说明：
  系统：ubuntu
  IDE: vscode
  gpu: GeForce RTX 2070

#### TIMIT数据集上的声纹识别实验

> TIMIT数据是收费的，但是国外有个大学网站提供了免费下载地:
[TIMIT数据下载地址](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3/tech&hit=1&filelist=1)
文件大小440M，是完整的数据库。
- 由于默认的目录名字和文件后缀都是大写的，而本项目只支持小写，可参考[shell递归遍历目录，修改文件名或者目录名](https://blog.csdn.net/qq_28228605/article/details/109963278)，将它们改为小写。

##### 1. 安装所需依赖包

```bash
$ pip install -r requirements.txt
```

##### 2.运行预处理文件`preprocess.py`
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

##### models目录下的模型介绍
- DeepSpeaker模型：[论文地址](https://arxiv.org/abs/1705.02304)
- Rest34 和 Rest50模型： [论文地址](https://arxiv.org/abs/1806.05622)
  
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
![1](/imgs/dpk_aucRoc.png)

##### VggVox模型实验结果
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
![1](/imgs/SEResNet_aucRoc.png)


  
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
![1](/imgs/SEResNet_aucRoc.png)

