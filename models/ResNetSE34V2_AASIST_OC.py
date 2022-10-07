import math
import torch
from ResNetModels.ResNetSE34V2 import MainModel
from aasist.models.AASIST import Model as AASISTModel
import json
from utils import load_parameters
import torch.nn as nn
import torch.nn.functional as F


class ASVModel(nn.Module):
    def __init__(self):
        super(ASVModel, self).__init__()
        
        resnet_weight = './ResNetModels/baseline_v2_ap.model' 
        model = MainModel()
        load_parameters(model.state_dict(), resnet_weight)
        
        self.model = model

        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()
            
        if True:
            #with torch.no_grad():
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, dim]
            emb_ASV = self.model(input_tmp)
        return emb_ASV

class CMModel(nn.Module):
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
            self.model.train()
            
        if True:
            #with torch.no_grad():
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, dim]
            emb_CM, _  = self.model(input_tmp)
        # print(emb_CM.shape)
        return emb_CM

class Model(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.asv_emb = ASVModel()
        self.cm_emb = CMModel()
        
        self.DNN_layer = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(negative_slope = 0.3),
            nn.Linear(512,256),
            nn.LeakyReLU(negative_slope = 0.3),
        )

        self.spf_fc = nn.Sequential(
                nn.Linear(160,512),
                nn.LeakyReLU(negative_slope = 0.3),
        )

        self.fc_out = torch.nn.Linear(256, 1, bias = False)
        # self.spf_out = torch.nn.Linear(512, 2, bias = False)

        conv_channels = [64, 128, 256]

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=conv_channels[0], bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=conv_channels[1], bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=conv_channels[2], bias=False, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.bn3 = nn.BatchNorm1d(conv_channels[2])

        self.relu = nn.LeakyReLU(negative_slope = 0.3)
    
    def forward(self, asv_enr_wav, asv_tst_wav):
        
        asv_enr_emb = self.asv_emb.extract_feat(asv_enr_wav.squeeze(-1))   # shape: (bs, 512)
        asv_tst_emb = self.asv_emb.extract_feat(asv_tst_wav.squeeze(-1))   # shape: (bs, 512)

        spf_emb = self.cm_emb.extract_feat(asv_tst_wav.squeeze(-1)) # shape: (bs, 160)
        spf_emb = self.spf_fc(spf_emb) # shape: (bs, 512)
        # pred_spf = self.spf_out(spf_emb)

        emb = torch.stack([asv_enr_emb, asv_tst_emb, spf_emb],dim=1) # shape: (bs, 3, 512)

        emb = self.relu(self.bn1(self.conv1(emb))) # shape: (bs, 64, 512)
        emb = self.relu(self.bn2(self.conv2(emb))) # shape: (bs, 128, 512)
        emb = self.relu(self.bn3(self.conv3(emb))) # shape: (bs, 256, 512)

        emb = F.adaptive_avg_pool1d(emb, output_size=4) # shape: (bs, 256, 4)
        emb = torch.flatten(emb, start_dim=1) # shape: (bs, 1024)        

        BB=self.DNN_layer(emb)
        out = self.fc_out(BB)

        return BB, out

    def validate(self, asv_enr_wav, asv_tst_wav):
        
        asv_enr_emb = self.asv_emb.extract_feat(asv_enr_wav.squeeze(-1))   # shape: (bs, 512)
        asv_tst_emb = self.asv_emb.extract_feat(asv_tst_wav.squeeze(-1))   # shape: (bs, 512)
        
        spf_emb = self.cm_emb.extract_feat(asv_tst_wav) # shape: (bs, 160)
        spf_emb = self.spf_fc(spf_emb) # shape: (bs, 512)
        # pred_spf = self.spf_out(spf_emb)

        emb = torch.stack([asv_enr_emb, asv_tst_emb, spf_emb],dim=1) # shape: (bs, 3, 512)

        emb = self.relu(self.bn1(self.conv1(emb))) # shape: (bs, 64, 512)
        emb = self.relu(self.bn2(self.conv2(emb))) # shape: (bs, 128, 512)
        emb = self.relu(self.bn3(self.conv3(emb))) # shape: (bs, 256, 512)

        emb = F.adaptive_avg_pool1d(emb, output_size=4) # shape: (bs, 256, 4)
        emb = torch.flatten(emb, start_dim=1) # shape: (bs, 1024) 

        BB=self.DNN_layer(emb)
        out = self.fc_out(BB)

        return BB, out