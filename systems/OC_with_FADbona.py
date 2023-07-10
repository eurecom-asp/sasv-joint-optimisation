import math
import os
import pickle as pk
from importlib import import_module
from typing import Any

import omegaconf
import pytorch_lightning as pl
import schedulers as lr_schedulers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import get_all_EERs
from utils import keras_decay
import json
from loss.loss import OCSoftmax


class System(pl.LightningModule):
    def __init__(
        self, config: omegaconf.dictconfig.DictConfig, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        _model = import_module("models.{}".format(config.model_arch))
        _model = getattr(_model, "Model")
        self.model = _model()
        self.configure_loss()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx, optimizer_idx):
        wave_asv_enr, wave_asv_tst, label = batch
        feats, lfcc_outputs = self.model(wave_asv_enr, wave_asv_tst)
        ocsoftmaxloss, _ = self.loss(feats, label)
        lfcc_loss = ocsoftmaxloss
        self.log(
            "trn_loss",
            lfcc_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return lfcc_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        embd_asv_enrol, wave_asv_tst, key, spkmd, utt_id, labels = batch
        feats, lfcc_outputs = self.model.validate(embd_asv_enrol, wave_asv_tst)
        _, pred = self.loss(feats, labels)
        return {"pred": pred, "key": key,"spkmd":spkmd,"utt_id":utt_id }

    def validation_epoch_end(self, outputs):
        log_dict = {}
        preds, keys,spkrs,utter_id = [], [], [], []
        for output in outputs:
            preds.append(output["pred"])
            keys.extend(list(output["key"]))
            spkrs.extend(list(output["spkmd"]))
            utter_id.extend(list(output["utt_id"]))
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)
        with open("exp_result/" + self.config.config_name + '/' + str(self.current_epoch) + '_dev.txt', '+a') as fh:
            for s, u, k, cm in zip(spkrs, utter_id,keys, preds):
                fh.write('{} {} {} {}\n'.format(s, u, k, cm))
        log_dict["sasv_eer_dev"] = sasv_eer
        log_dict["sv_eer_dev"] = sv_eer
        log_dict["spf_eer_dev"] = spf_eer

        self.log_dict(log_dict)

    def test_step(self, batch, batch_idx, dataloader_idx=-1):
        res_dict = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        return res_dict

    def test_epoch_end(self, outputs):
        log_dict = {}
        preds, keys,spkrs,utter_id = [], [], [], []
        for output in outputs:
            preds.append(output["pred"])
            keys.extend(list(output["key"]))
            spkrs.extend(list(output["spkmd"]))
            utter_id.extend(list(output["utt_id"]))
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)
        with open("exp_result/" + self.config.config_name + '/eval.txt', '+a') as fh:
            for s, u, k, cm in zip(spkrs, utter_id,keys, preds):
                fh.write('{} {} {} {}\n'.format(s, u, k, cm))
        log_dict["sasv_eer_eval"] = sasv_eer
        log_dict["sv_eer_eval"] = sv_eer
        log_dict["spf_eer_eval"] = spf_eer

        self.log_dict(log_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.wd,
        )
        optimizer_oc = torch.optim.Adam(
            params=self.loss.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.wd,
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.config.optim.gamma,
        )
        lr_scheduler_oc = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_oc, gamma=self.config.optim.gamma,
        )
        return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 200,
                "strict": True,
            },
        },
        {
            "optimizer": optimizer_oc, 
            "lr_scheduler": {
                "scheduler": lr_scheduler_oc,
                "interval": "step",
                "frequency": 200,
                "strict": True,
            },
            },
    )

    def setup(self, stage=None):
        """
        configures dataloaders.

        Args:
            stage: one among ["fit", "validate", "test", "predict"]
        """
        self.load_meta_information()

        if stage == "fit" or stage is None:
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_trn = getattr(module, "get_trnset")
            self.ds_func_dev = getattr(module, "get_dev_evalset")
        elif stage == "validate":
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_dev = getattr(module, "get_dev_evalset")
        elif stage == "test":
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_eval = getattr(module, "get_evalset")
        else:
            raise NotImplementedError(".....")

    def train_dataloader(self):
        self.train_ds = self.ds_func_trn(self.cm_embd_trn, 'train', self.spk_meta_trn_withFAD, self.spk_meta_trn, self.config.dirs.database)
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.loader.n_workers,
        )

    def val_dataloader(self):
        with open(self.config.dirs.sasv_dev_trial, "r") as f:
            sasv_dev_trial = f.readlines()
        utt2spk_dev = {}
        
        with open(self.config.dirs.sasv_dev_trn_trial, "r") as f:
            sasv_dev_trn_trial = f.readlines()
            for line in sasv_dev_trn_trial:
                tmp = line.strip().split(" ")
                spk = tmp[0]
                utts = tmp[1]
                utt2spk_dev[spk] = utts
                
        self.dev_ds = self.ds_func_dev(
            sasv_dev_trial, utt2spk_dev, 'dev', self.config.dirs.database)
        return DataLoader(
            self.dev_ds,
            batch_size=self.config.batch_size//5,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
        )

    def test_dataloader(self):
        utt2spk = {}
        with open(self.config.dirs.sasv_eval_trial, "r") as f:
            sasv_eval_trial = f.readlines()
        with open(self.config.dirs.sasv_eval_trn_trial, "r") as f:
            sasv_eval_trn_trial = f.readlines()
            for line in sasv_eval_trn_trial:
                tmp = line.strip().split(" ")
                spk = tmp[0]
                utts = tmp[1]
                utt2spk[spk] = utts
        self.eval_ds = self.ds_func_eval(
            sasv_eval_trial, utt2spk, 'eval', self.config.dirs.database)
        return DataLoader(
            self.eval_ds,
            batch_size=self.config.batch_size//7,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
        )

    def configure_loss(self):
        self.loss = OCSoftmax(feat_dim=self.config.optim.oc_dim, 
                                r_real=self.config.optim.r_real, 
                                r_fake=self.config.optim.r_fake, 
                                alpha=self.config.optim.alpha)  


    def load_meta_information(self):
        with open(self.config.dirs.spk_meta + "spk_meta_trn.pk", "rb") as f:
            self.spk_meta_trn = pk.load(f)
        with open(self.config.dirs.embedding + "cm_embd_trn.pk", "rb") as f:
            self.cm_embd_trn = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_trn_withFAD.pk", "rb") as f:
            self.spk_meta_trn_withFAD = pk.load(f)