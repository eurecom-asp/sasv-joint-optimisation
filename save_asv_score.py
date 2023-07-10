import argparse
import json
import os
import pickle as pk
from pathlib import Path
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

database =  "/path/to/your/LA/"

# list of dataset partitions
SET_PARTITION = ["trn", "dev", "eval"]

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

def get_cos_score(set_name, config_name):
    with open(config_name + "/asv_embd_" + set_name + ".pk", "rb") as f:
        asv_embs = pk.load(f)
    with open(config_name + "/spk_model_" + set_name + ".pk", "rb") as f:
        spk_models = pk.load(f)
    protocol = SET_SASV_PROTOCOL[set_name]
    meta_lines = open(protocol, "r").readlines()
    preds, keys = [], []
    for line in meta_lines:
        tmp = line.strip().split(" ")
        spk = tmp[0]
        utts = tmp[1]
        spk_model = spk_models[spk]
        embd = asv_embs[utts]
        cosine = np.dot(spk_model,embd)/(norm(spk_model)*norm(embd))
        keys.append(utts)
        preds.append(cosine)
    with open(config_name + "/" + "score_ASV_" + set_name + '.txt', '+a') as fh:
        for k,cm in zip(keys, preds):
            fh.write('{} {} \n'.format(k,cm))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comment", type=str, default="Exp-SASV"
    )

    args = parser.parse_args()

    config_name = args.comment

    for set_name in SET_PARTITION:
        if set_name == "trn":
            continue
        get_cos_score(set_name, config_name)