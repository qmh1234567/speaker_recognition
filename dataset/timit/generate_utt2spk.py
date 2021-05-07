import pickle
import glob
import os


test_pk_dir = '/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/test/'

ftest = './utt2spk'

def create_utt2Spk(test_pk_dir):
    
    audio_paths = [pickle for pickle in glob.iglob(test_pk_dir +"/*.pickle")]
        
    audio_paths.sort()
        
    audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in audio_paths]
    
    with open(ftest,'w') as f:
        for i in range(0,len(audio_labels)):
            line = audio_paths[i]+' '+audio_labels[i]
            f.write(line+"\n")
    
if __name__ == "__main__":
    create_utt2Spk(test_pk_dir)