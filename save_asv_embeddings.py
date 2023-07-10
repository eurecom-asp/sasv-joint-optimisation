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


from utils import load_parameters


database =  "/path/to/your/LA/"


# list of dataset partitions
SET_PARTITION = ["trn", "dev", "eval"]

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


def save_embeddings(
    set_name, cm_embd_ext, asv_embd_ext, device, config_name
):
    meta_lines = open(SET_CM_PROTOCOL[set_name], "r").readlines()
    utt2spk = {}
    utt_list = []
    for line in meta_lines:
        tmp = line.strip().split(" ")

        spk = tmp[0]
        utt = tmp[1]

        if utt in utt2spk:
            print("Duplicated utt error", utt)

        utt2spk[utt] = spk
        utt_list.append(utt)
    print(set_name)
    print(SET_DIR[set_name])
    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=90, shuffle=False, drop_last=False, pin_memory=True
    )

    cm_emb_dic = {}
    asv_emb_dic = {}

    print("Getting embedgins from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            # print("cm", batch_x.shape)
            batch_cm_emb, _ = cm_embd_ext.extract_feat(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            batch_asv_emb = asv_embd_ext.extract_feat(batch_x).detach().cpu().numpy()

        for k, cm_emb, asv_emb in zip(key, batch_cm_emb, batch_asv_emb):
            cm_emb_dic[k] = cm_emb
            asv_emb_dic[k] = asv_emb

    os.makedirs(config_name, exist_ok=True)
    with open( config_name + "/cm_embd_%s.pk" % (set_name), "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open(config_name + "/asv_embd_%s.pk" % (set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)


def save_models(set_name, asv_embd_ext, device, config_name):
    utt2spk = {}
    utt_list = []

    for trn in SET_TRN[set_name]:
        meta_lines = open(trn, "r").readlines()

        for line in meta_lines:
            tmp = line.strip().split(" ")

            spk = tmp[0]
            utts = tmp[1].split(",")

            for utt in utts:
                if utt in utt2spk:
                    print("Duplicated utt error", utt)

                utt2spk[utt] = spk
                utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = Dataset_ASVspoof2019_devNeval(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )
    asv_emb_dic = {}

    print("Getting embedgins from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            # print("asv", batch_x.shape)
            batch_asv_emb = asv_embd_ext.extract_feat(batch_x).detach().cpu().numpy()

        for k, asv_emb in zip(key, batch_asv_emb):
            utt = k
            spk = utt2spk[utt]

            if spk not in asv_emb_dic:
                asv_emb_dic[spk] = []

            asv_emb_dic[spk].append(asv_emb)

    for spk in asv_emb_dic:
        asv_emb_dic[spk] = np.mean(asv_emb_dic[spk], axis=0)

    with open(config_name + "/spk_model_%s.pk" % (set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)


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
        if set_name == "trn":
            continue
        save_models(set_name, asv_embd_ext, device, config_name)