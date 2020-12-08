# libri
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean  --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean/pickle  --data_type libri

# voxceleb
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Voxceleb/vox1_dev_wav/ --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Voxceleb/vox1_dev_wav/pickle --data_type vox1
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Voxceleb/vox_test_wav/ --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Voxceleb/vox_test_wav/pickle --data_type vox1

# mit
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TRAIN/wav --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TRAIN/pickle --data_type mit
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TEST/wav --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TEST/pickle --data_type mit

python newpreprocess.1.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean  --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean/pickle_fft  --data_type libri

python newpreprocess.1.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/train-clean-100/LibriSpeech/train-clean-100  --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/train-clean-100/LibriSpeech/train-clean-100/pickle_fft1  --data_type libri