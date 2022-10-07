import argparse
import json
import os
import pickle as pk
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aasist.data_utils import Dataset_ASVspoof2019_devNeval, Dataset_ASVspoof2019_train
from aasist.models.AASIST import Model as AASISTModel
# from ECAPATDNN.model import ECAPA_TDNN
from ResNetModels.ResNetSE34V2 import MainModel
from models.ResNetSE34V2_AASIST_OC import Model
from dataloaders.Datalaoder_sasv import get_evalset
from loss.loss import OCSoftmax

from utils import load_parameters

# list of dataset partitions
# SET_PARTITION = ["dev"]
SET_PARTITION = ["eval"]

database =  "/path/to/your/LA/"

# directories of each dataset partition
SET_DIR = {
    "trn": database + "ASVspoof2019_LA_train/",
    "dev": database + "ASVspoof2019_LA_dev/",
    "eval": database + "ASVspoof2019_LA_eval/",
}

sasv_eval_trial = "./protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
sasv_eval_trn_trial = "./protocols/ASVspoof2019.LA.asv.eval.gi.trn.txt"

utt2spk = {}
with open(sasv_eval_trial, "r") as f:
    sasv_eval_trial = f.readlines()
with open(sasv_eval_trn_trial, "r") as f:
    sasv_eval_trn_trial = f.readlines()
    for line in sasv_eval_trn_trial:
        tmp = line.strip().split(" ")
        spk = tmp[0]
        utts = tmp[1]
        utt2spk[spk] = utts
eval_ds = get_evalset(sasv_eval_trial, utt2spk, 'eval', database)

def save_embeddings(
    set_name, model, loss, device, config_name
):
    
    loader = DataLoader(
        eval_ds, batch_size=10, shuffle=False, drop_last=False, pin_memory=True
    )
    preds, keys, spkrs, utter_id = [], [], [], []
    for wave_asv_enr, wave_asv_tst, key, spkmd, utt_id, labels in tqdm(loader):
        wave_asv_enr = wave_asv_enr.to(device)
        wave_asv_tst = wave_asv_tst.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            feats, lfcc_outputs = model.validate(wave_asv_enr, wave_asv_tst)
            _, pred = loss(feats, labels)
            preds.append(pred)
            keys.extend(list(key))
            spkrs.extend(list(spkmd))
            utter_id.extend(list(utt_id))

    preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    os.makedirs(config_name, exist_ok=True)
    with open(config_name + "/" + "score_SASV_" + set_name + '.txt', '+a') as fh:
        for s, u, k, cm in zip(spkrs, utter_id, keys, preds):
            fh.write('{} {} {} {}\n'.format(s, u, k, cm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-aasist_config", type=str, default="./aasist/config/AASIST.conf"
    )
    parser.add_argument(
        "-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth"
    )
    parser.add_argument(
        "--model", type=str, default="./pre_trained_models/ResNetSE34V2_AASIST_OC.ckpt"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    PATH = args.model
    # address = PATH.split('/')
    # config_name = address[9]
    config_name = 'EXP-score'

    with open(args.aasist_config, "r") as f_json:
        config = json.loads(f_json.read())

    loss = OCSoftmax(feat_dim=256, r_real=0.8, r_fake=0.2, alpha=10.0)  
    model = Model()
    checkpoint = torch.load(PATH)
    dicts = {}
    loss_dict = {}
    for k in checkpoint['state_dict']:
        if k[:6] == 'model.':
            k_ = k[6:] 
            dicts[k_] = checkpoint['state_dict'][k]
        loss_dict['center'] = checkpoint['state_dict']['loss.center']
    model.load_state_dict(dicts)
    model.to(device)
    model.eval()
    loss.load_state_dict(loss_dict)
    loss.to(device)
    loss.eval()

    for set_name in SET_PARTITION:
        save_embeddings(
            set_name,
            model,
            loss,
            device,
            config_name,
        )