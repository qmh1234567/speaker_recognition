import tensorflow as tf
import re
import os
import glob
import sys
import pickle
import random
import numpy as np
import argparse
from python_speech_features import logfbank
import vad_ex
import webrtcvad
from progress.bar import Bar
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import wave
import logging
import math


# class SilenceDetector(object):
#     def __init__(self, threshold=20, bits_per_sample=16):
#         self.cur_SPL = 0
#         self.threshold = threshold
#         self.bits_per_sample = bits_per_sample
#         self.normal = pow(2.0, bits_per_sample - 1)
#         self.logger = logging.getLogger('balloon_thrift')

#     def is_silence(self, chunk):
#         self.cur_SPL = self.soundPressureLevel(chunk)
#         is_sil = self.cur_SPL < self.threshold
#         # print('cur spl=%f' % self.cur_SPL)
#         if is_sil:
#             self.logger.debug('cur spl=%f' % self.cur_SPL)
#         return is_sil
    
#     def soundPressureLevel(self, chunk):
#         value = math.pow(self.localEnergy(chunk), 0.5)
#         value = value / len(chunk) + 1e-12
#         value = 20.0 * math.log(value, 10)
#         return value

#     def localEnergy(self, chunk):
#         power = 0.0
#         for i in range(len(chunk)):
#             sample = chunk[i] * self.normal
#             power += sample*sample
#         return power


class Preprocess():
    def __init__(self, hparams):
        # Set hparams
        self.hparams = hparams
    def preprocess_data(self):
        if self.hparams.data_type == "libri":
            path_list = [x for x in glob.iglob(
                self.hparams.in_dir.rstrip("/")+"/*/*/*.wav")]
        elif self.hparams.data_type == "vox1":
            path_list = [x for x in glob.iglob(
                self.hparams.in_dir.rstrip("/")+"/wav/*/*/*.wav")]
        elif self.hparams.data_type == 'vox2':
            path_list = [x for x in glob.iglob(
                self.hparams.in_dir.rstrip("/")+"/wav/*/*/*.m4a")]
        elif self.hparams.data_type == 'mit':
            path_list = [x for x in glob.iglob(
                self.hparams.in_dir.rstrip("/")+"/*/*/*.wav")]
            print(len(path_list))
        else:
            raise ValueError("data type not supported")

        bar = Bar("Processing", max=(len(path_list)),
                  fill='#', suffix='%(percent)d%%')
        # 对每个音频进行预处理
        for path in path_list:
            bar.next()
            # 去静音
            wav_arr, sample_rate = self.vad_process(path)
            # signal,sample_rate = librosa.load(path,16000)
            # wav_arr = self.vad_audio(signal,sample_rate)
            if sample_rate != 16000:
                print("sample rate do meet the requirement")
                exit()
            # padding 音频裁减
            wav_arr = self.cut_audio(wav_arr,sample_rate)
            # 提取特征并保存
            self.create_pickle(path, wav_arr, sample_rate)
        bar.finish()
    
    # 裁减音频
    def cut_audio(self,wav_arr,sample_rate):
        singal_len = int(self.hparams.segment_length*sample_rate)
        n_sample = wav_arr.shape[0]
        if n_sample < singal_len:
                wav_arr = np.hstack((wav_arr, np.zeros(singal_len-n_sample)))
        else:
            wav_arr = wav_arr[(n_sample-singal_len) //2:(n_sample+singal_len)//2]
        return wav_arr
    
    # remove VAD
    def vad_audio(self,wav,sr,threshold = 10):
        sil_detector = SilenceDetector(threshold)
        new_wav = []
        if sr != 16000:
            wav = librosa.resample(wav, sr, 16000)
            sr = 16000
        for i in range(int(len(wav)/(sr*0.02))):
            start = int(i*sr*0.02)
            end = start + int(sr*0.02)
            is_silence = sil_detector.is_silence(wav[start:end])
            if not is_silence:
                new_wav.extend(wav[start:end])
        return np.array(new_wav)
    
    
    # VAD去静音
    def vad_process(self, path):
        # VAD Process
        if self.hparams.data_type == "vox1":
            audio, sample_rate = vad_ex.read_wave(path)
        elif self.hparams.data_type == "vox2":
            audio, sample_rate = vad_ex.read_m4a(path)
        elif self.hparams.data_type == "libri":
            audio, sample_rate = vad_ex.read_libri(path)
        elif self.hparams.data_type == 'mit':
            audio, sample_rate = vad_ex.read_libri(path)
        vad = webrtcvad.Vad(1)
        frames = vad_ex.frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
        total_wav = b""
        for i, segment in enumerate(segments):
            total_wav += segment
        # Without writing, unpack total_wav into numpy [N,1] array
        wav_arr = np.frombuffer(total_wav, dtype=np.int16)
        # print("read audio data from byte string. np array of shape:"+str(wav_arr.shape))
        return wav_arr, sample_rate
    
    
    def plot_spectrogram(self,spec,ylabel):
        fig = plt.figure()
        heatmap = plt.pcolor(spec)
        fig.colorbar(mappable=heatmap)
        plt.xlabel('Time(s)')
        plt.ylabel(ylabel)
        plt.tight_layout()
        # plt.show()
    
    # 提取fbank特征
    def extract_feature(self,wav_arr,sample_rate,path):
        save_dict = {}
        logmel_feats = logfbank(
            wav_arr, samplerate=sample_rate, nfilt=self.hparams.spectrogram_scale)
        save_dict["LogMel_Features"] = logmel_feats
        # self.plot_spectrogram(logmel_feats.T,'Filter Banks')
        return save_dict    
    
    
    ## 写入pickle文件
    def create_pickle(self, path, wav_arr, sample_rate):
        if round((wav_arr.shape[0] / sample_rate), 1) >= self.hparams.segment_length:
            # 提取特征
            save_dict = self.extract_feature(wav_arr,sample_rate,path)
            
            if self.hparams.data_type == "vox1" or self.hparams.data_type == "vox2":
                data_id = "_".join(path.split("/")[-3:])
                save_dict["SpkId"] = path.split("/")[-3]
                save_dict["ClipId"] = path.split("/")[-2]
                save_dict["WavId"] = path.split("/")[-1]
                if self.hparams.data_type == "vox1":
                    pickle_f_name = data_id.replace("wav", "pickle")
                elif self.hparams.data_type == "vox2":
                    pickle_f_name = data_id.replace("m4a", "pickle")

            elif self.hparams.data_type == "libri":
                # data_id = "_".join(path.split("/")[-3:])
                data_id = path.split("/")[-1].replace("-", "_")  # 音频格式 5514_19192_0011.wav
                pickle_f_name = data_id.replace("wav", "pickle")
            
            elif self.hparams.data_type == 'mit':
                data_id = "_".join(path.split("/")[-2:])
                save_dict["SpkId"] = path.split("/")[-2]
                pickle_f_name = data_id.replace("wav", "pickle")

            if not os.path.exists(self.hparams.pk_dir):
                os.mkdir(self.hparams.pk_dir)
            with open(self.hparams.pk_dir + "/" + pickle_f_name, "wb") as f:
                pickle.dump(save_dict, f, protocol=3)
        else:
            print("wav length smaller than 1.6s: " + path)

    # 绘制音频
    def draw_waveform(self,wav_arr,sample_rate):
        plt.figure()
        librosa.display.waveplot(wav_arr,sample_rate)
    
        
    # 绘制语谱图
    def draw_spectrum(self,filename):
        f = wave.open(filename,'rb')
        # 得到语音参数
        params = f.getparams()
        nchannels, sampwidth, framerate,nframes = params[:4]
        # 得到的数据是字符串，需要将其转成int型
        strData = f.readframes(nframes)
        wavaData = np.fromstring(strData,dtype=np.int16)
        # 归一化
        wavaData = wavaData * 1.0/max(abs(wavaData))
        # .T 表示转置
        wavaData = np.reshape(wavaData,[nframes,nchannels]).T
        f.close()
        # 绘制频谱
        plt.specgram(wavaData[0],Fs = framerate,scale_by_freq=True,sides='default')
        plt.ylabel('Frequency')
        plt.xlabel('Time(s)')
        # plt.show()



    def test_singleAudio(self,path):
        self.draw_spectrum(path)
        signal,sample_rate = librosa.load(path,16000)
        self.draw_waveform(signal,sample_rate)  # 预处理前
        # 去静音
        # wav_arr = self.vad_audio(signal,sample_rate)
        wav_arr, sample_rate = self.vad_process(path)
        self.draw_waveform(wav_arr,sample_rate)  # 去静音后
        # plt.show()
        # exit()
        if sample_rate != 16000:
            print("sample rate do meet the requirement")
            exit()
        # padding 音频裁减
        wav_arr = self.cut_audio(wav_arr,sample_rate)
        # self.draw_waveform(wav_arr,sample_rate)  # 预处理后
        # self.draw_spectrum(wav_arr,sample_rate)
        # 提取特征并保存
        self.create_pickle(path, wav_arr, sample_rate)
        # self.draw_spectrum(wav_arr,sample_rate)
    
        plt.show()
    
def main():

    # Hyperparameters

    parser = argparse.ArgumentParser()

    # timit
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/TIMIT_M/TIMIT/train/ --pk_dir=/home/qmh/Projects/Datasets/TIMIT_M/train/ --data_type=mit
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/TIMIT_M/TIMIT/test/ --pk_dir=/home/qmh/Projects/Datasets/TIMIT_M/test/ --data_type=mit
    # libri    
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/LibriSpeech/train-clean-100/ --pk_dir=/home/qmh/Projects/Datasets/LibriSpeech_O/train-clean-100/ --data_type=libri
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/LibriSpeech/test-clean/ --pk_dir=/home/qmh/Projects/Datasets/LibriSpeech_O/test-clean/ --data_type=libri
    parser.add_argument("--in_dir", type=str, required=True,
                        help="input audio data dir")
    parser.add_argument("--pk_dir", type=str, required=True,
                        help="output pickle dir")
    parser.add_argument("--data_type", required=True,
                        choices=["libri", "vox1", "vox2","mit"])

    # Data Process
    parser.add_argument("--segment_length", type=float,
                        default=3, help="segment length in seconds")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                        help="scale of the input spectrogram")
    args = parser.parse_args()

    preprocess = Preprocess(args)

    preprocess.preprocess_data()
    # path = "/home/qmh/Projects/Datasets/TIMIT_M/TIMIT/train/dr4/fcag0/sx153.wav"
    # path = "/home/qmh/Projects/Datasets/LibriSpeech/train-clean-100/200/126784/200-126784-0025.wav"
    # preprocess.test_singleAudio(path)


if __name__ == "__main__":
    main()
