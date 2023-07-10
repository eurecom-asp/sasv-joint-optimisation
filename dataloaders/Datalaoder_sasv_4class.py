import random
from typing import Dict, List
import numpy as np
from torch.utils.data import Dataset
import librosa
from torch import Tensor
import soundfile as sf
import pickle as pk

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

def read_waveform(path):
    cut = 64600
    X, _ = sf.read(path)
    X_pad = pad_random(X, cut)
    x_inp = Tensor(X_pad)
    return x_inp

def read_waveform_eval(path):
    cut = 64600
    X, _ = sf.read(path)
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    return x_inp

def read_waveform_eval_1(path):
    cut = 960000
    X, _ = sf.read(path)
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    return x_inp

def read_waveform_eval_2(path):
    cut = 1440000
    # cut = 960000
    X, _ = sf.read(path)
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    return x_inp

class SASV_Trainset(Dataset):
    def __init__(self, cm_embd, partition, spk_meta, data_dir):

        if partition == 'train':
            self.feature_address = data_dir + 'ASVspoof2019_LA_train/flac/'
        elif partition == 'dev':
            self.feature_address = data_dir + 'ASVspoof2019_LA_dev/flac/'
        elif partition == 'eval':
            self.feature_address = data_dir + 'ASVspoof2019_LA_eval/flac/' 

        with open('./spk_meta/trn_spk.pk', "rb") as f:
            self.speaker_id = pk.load(f)
        
        self.cm_embd = cm_embd
        self.spk_meta = spk_meta

        print("-----returning CM and ASV-----")

    def __len__(self):
        return len(self.cm_embd.keys())

    def __getitem__(self, index):

        randnumber = random.randint(0, 3)
        ans_type = 1
        if randnumber == 0:  # target bonafide
            spk = random.choice(list(self.spk_meta.keys()))
            enr, tst = random.sample(self.spk_meta[spk]["bonafide"], 2)
            ans_type = 0

        elif randnumber == 1:  # nontarget bonafide
            spk, ze_spk = random.sample(self.spk_meta.keys(), 2)
            enr = random.choice(self.spk_meta[spk]["bonafide"])
            tst = random.choice(self.spk_meta[ze_spk]["bonafide"])

        elif randnumber == 2:  # target spoof
            spk = random.choice(list(self.spk_meta.keys()))
            if len(self.spk_meta[spk]["spoof"]) == 0:
                while True:
                    spk = random.choice(list(self.spk_meta.keys()))
                    if len(self.spk_meta[spk]["spoof"]) != 0:
                        break
            enr = random.choice(self.spk_meta[spk]["bonafide"])
            tst = random.choice(self.spk_meta[spk]["spoof"])
            
        elif randnumber == 3: # nontarget spoof
            spk, ze_spk = random.sample(self.spk_meta.keys(), 2)
            enr = random.choice(self.spk_meta[spk]["bonafide"])
            tst = random.choice(self.spk_meta[ze_spk]["spoof"])
                
        wave_asv_enr = read_waveform_eval(self.feature_address + enr + '.flac')
        wave_asv_tst = read_waveform_eval(self.feature_address + tst + '.flac')
        
        return wave_asv_enr, wave_asv_tst, ans_type


class SASV_DevEvalset(Dataset):
    def __init__(self, utt_list, spk_model, partition, data_dir):
        self.utt_list = utt_list
        self.spk_model = spk_model
        if partition == 'dev':
            self.feature_address = data_dir + 'ASVspoof2019_LA_dev/flac/'
            self.feature_address_ern = './enr_audio/dev/'
        elif partition == 'eval':
            self.feature_address = data_dir + 'ASVspoof2019_LA_eval/flac/' 
            self.feature_address_ern = './enr_audio/eval/'

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, index):
        line = self.utt_list[index]
        
        spkmd, key, _, ans = line.strip().split(" ")
        
        wave_asv_tst = read_waveform_eval(self.feature_address + key + '.flac')
        
        #enrolment
        enr_wav=self.spk_model[spkmd]
       
        wave_asv_enr=read_waveform_eval_1(self.feature_address_ern + enr_wav + '.flac')

        if ans == 'target':
            label = 0
        else:
            label = 1
        
        return wave_asv_enr, wave_asv_tst, ans, spkmd, key, label


class SASV_Evalset(Dataset):
    def __init__(self, utt_list, spk_model, partition, data_dir):
        self.utt_list = utt_list
        self.spk_model = spk_model
        if partition == 'dev':
            self.feature_address = data_dir + 'ASVspoof2019_LA_dev/flac/'
            self.feature_address_ern = './enr_audio/dev/'
        elif partition == 'eval':
            self.feature_address = data_dir + 'ASVspoof2019_LA_eval/flac/' 
            self.feature_address_ern = './enr_audio/eval/'

    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, index):
        line = self.utt_list[index]
        
        spkmd, key, _, ans = line.strip().split(" ")
        
        wave_asv_tst = read_waveform_eval(self.feature_address + key + '.flac')
        

        #enrolment
        enr_wav=self.spk_model[spkmd]
       
        wave_asv_enr=read_waveform_eval_2(self.feature_address_ern + enr_wav + '.flac')
        if ans == 'target':
            label = 0
        else:
            label = 1
        
        return wave_asv_enr, wave_asv_tst, ans, spkmd, key, label


def get_trnset(
    cm_embd_trn: Dict, partition_flag: Dict, spk_meta_trn: Dict, database: Dict,
) -> SASV_DevEvalset:
    return SASV_Trainset(
        cm_embd=cm_embd_trn, partition=partition_flag, spk_meta=spk_meta_trn, data_dir=database,
    )

def get_dev_evalset(
    utt_list: List, spk_model: Dict, partition_flag: Dict, database: Dict,
) -> SASV_DevEvalset:
    return SASV_DevEvalset(
        utt_list=utt_list, spk_model=spk_model, partition=partition_flag, data_dir=database,
    )


def get_evalset(
    utt_list: List, spk_model: Dict, partition_flag: Dict, database: Dict,
) -> SASV_Evalset:
    return SASV_Evalset(
        utt_list=utt_list, spk_model=spk_model, partition=partition_flag, data_dir=database,
    )