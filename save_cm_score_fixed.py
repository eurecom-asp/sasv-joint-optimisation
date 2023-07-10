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
from models.ResNetSE34V2_AASIST_OC_fixed import Model
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

SET_SASV_PROTOCOL = {
    "dev": "protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt",
    "eval": "protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt",
}

sasv_eval_trial = "./protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt"
sasv_eval_trn_trial = "./protocols/ASVspoof2019.LA.asv.eval.gi.trn.txt"

class CMModel():
    def __init__(self):
        super(CMModel, self).__init__()
        
        aasist_weight = './aasist/models/weights/AASIST.pth' 
        with open('./aasist/config/AASIST.conf', "r") as f_json:
            config = json.loads(f_json.read())
        aasist_model_config = config["model_config"]
        model = AASISTModel(aasist_model_config)
        load_parameters(model.state_dict(), aasist_weight)
        
        self.model = model

        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.eval()
            
        # if True:
        with torch.no_grad():
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, dim]
            emb_CM, score_CM  = self.model(input_tmp)
        # print(emb_CM.shape)
        return emb_CM, score_CM


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
    set_name, cm_embd_ext, device, config_name
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
        "--comment", type=str, default="Exp-SASV"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))


    cm_embd_ext= CMModel()

    config_name = args.comment

    for set_name in SET_PARTITION:
        save_embeddings(
            set_name,
            cm_embd_ext,
            device,
            config_name,
        )