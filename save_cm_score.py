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

from utils import load_parameters

database =  "/path/to/your/LA/"


# list of dataset partitions
SET_PARTITION = ["dev", "eval"]

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "trn": "protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

# directories of each dataset partition
SET_DIR = {
    "trn": database + "ASVspoof2019_LA_train/",
    "dev": database + "ASVspoof2019_LA_dev/",
    "eval": database + "ASVspoof2019_LA_eval/",
}

# enrolment data list for speaker model calculation
# each speaker model comprises multiple enrolment utterances
SET_TRN = {
    "dev": [
        database + "/ASVspoof2019.LA.asv.dev.female.trn.txt",
        database + "/ASVspoof2019.LA.asv.dev.male.trn.txt",
    ],
    "eval": [
        database + "/ASVspoof2019.LA.asv.eval.female.trn.txt",
        database + "/ASVspoof2019.LA.asv.eval.male.trn.txt",
    ],
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
    set_name, cm_embd_ext, asv_embd_ext, device, config_name
):
    

    print("Getting embedgins from set %s..." % (set_name))
    preds, keys = [], []
    preds_softmax = []
    i = 0
    loader = DataLoader(
        eval_ds, batch_size=50, shuffle=False, drop_last=False, pin_memory=True
    )
    for _, batch_x, _, _, key, _ in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_cm_score = cm_embd_ext.extract_feat(batch_x)
            batch_cm_score_sofxmax = torch.nn.functional.softmax(batch_cm_score, dim = 1)
            batch_cm_score = batch_cm_score[:,1]
            batch_cm_score_sofxmax = batch_cm_score_sofxmax[:,1]
            batch_cm_score = batch_cm_score.detach().cpu().numpy()
            batch_cm_score_sofxmax = batch_cm_score_sofxmax.detach().cpu().numpy()
        keys.extend(key)
        preds.extend(batch_cm_score.tolist())
        preds_softmax.extend(batch_cm_score_sofxmax.tolist())
        
    os.makedirs(config_name, exist_ok=True)
    with open(config_name + '/' + 'score_CM_eval_no_softmax.txt', '+a') as fh:
        for k,cm in zip(keys, preds):
            fh.write('{} {} \n'.format(k,cm))
    with open(config_name + '/' + 'score_CM_eval.txt', '+a') as fh:
        for k,cm in zip(keys, preds_softmax):
            fh.write('{} {} \n'.format(k,cm))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str,
    )
    parser.add_argument(
        "--comment", type=str, default="Exp-SASV"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    model = Model()
    PATH = args.model
    checkpoint = torch.load(PATH)
    dicts = {}
    for k in checkpoint['state_dict']:
        if k[:6] == 'model.':
            k_ = k[6:] 
            dicts[k_] = checkpoint['state_dict'][k]
    model.load_state_dict(dicts)
    model.to(device)
    model.eval()
    cm_embd_ext = model.cm_emb
    asv_embd_ext = model.asv_emb

    config_name = args.comment

    for set_name in SET_PARTITION:
        save_embeddings(
            set_name,
            cm_embd_ext,
            asv_embd_ext,
            device,
            config_name,
        )