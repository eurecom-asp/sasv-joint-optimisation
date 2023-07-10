# sasv-joint-optimisation 

This repository provides codes to reproduce our papers on joint optimisation of speaker verification and spoofing countermeasure systems.

The corresponding papers are [On the potential of jointly-optimised solutions to spoofing attack detection and automatic speaker verification](https://arxiv.org/pdf/2209.00506.pdf) accepted to IberSPEECH 2022 and [Can spoofing countermeasure and speaker verification systems be jointly optimised?](https://arxiv.org/pdf/2303.07073.pdf) accepted to ICASSP 2023.

### Notes

~~Codes to our ICASSP 2023 paper "Can spoofing countermeasure and speaker verification systems be jointly optimised?" will be uploaded soon.~~

(July 10, 2023) We've uploaded the <strong>pre-trained models and score files</strong> used in our experiment in [here](https://nextcloud.eurecom.fr/s/84zF6XEDsXFjWGo). Please notice that due to storage limit, only the best performing model of each configuration is uploaded.

(July 10, 2023, two hours later) We've uploaded the codes for the ICASSP paper. For experiments related to FAD database, please first download the data from [here](https://zenodo.org/record/6623227), convert all .wav files under `'FAD/train/real/aishell3/'` and `'FAD/train/fake/'` to .flac and copy them to `'/path/to/your/LA/ASVspoof2019_LA_train/flac/'`.

### Getting started
Codes were tested using a GeForce RTX 3090 GPU with CUDA Version==11.2. Please try to install the dependencies first:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

```
Then:
```
pip install -r requirements.txt
```
### Dataset
The ASVspoof 2019 database can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336).

Please change the `database` defined in each `.conf` file from `'/path/to/your/LA/'` to its actual path.

### Training 
To jointly-optimise the ASV and CM sub-systems:
```
python main.py --config ./configs/ResNetSE34V2_AASIST_OC.conf
```
To train a back-end classifier and keep ASV and CM fixed:
```
python main.py --config ./configs/ResNetSE34V2_AASIST_OC_fixed.conf
```

### Pre-trained models

You can try to calculate the evaluation score by:
```
python save_sasv_score.py
```
And for the scores of ASV sub-system, please first try:
```
python save_asv_embeddings.py --model=path/to/your/pre_trained_model.ckpt --comment="conf1"
```
then
```
python save_asv_score.py --comment="conf1"
```
For the scores of CM sub-system, please try:
```
python save_cm_score.py --model=path/to/your/pre_trained_model.ckpt --comment="conf1"
```

### References
If you find this repository useful, please consider citing:
```
@inproceedings{sasv_joint_vol1,
  author={Wanying Ge and Hemlata Tak and Massimiliano Todisco and Nicholas Evans},
  title={{On the potential of jointly-optimised solutions to spoofing attack detection and automatic speaker verification}},
  year=2022,
  booktitle={Proc. IberSPEECH 2022},
  pages={51--55},
}
```
and
```
@inproceedings{sasv_joint_vol2,
  author={Wanying Ge and Hemlata Tak and Massimiliano Todisco and Nicholas Evans},
  title={{Can spoofing countermeasure and speaker verification systems be jointly optimised?}},
  year=2023,
  booktitle={Proc. ICASSP 2023},
}
```


### Acknowledgements
Codes are based on the implementations of [SASVC2022_Baseline](https://github.com/sasv-challenge/SASVC2022_Baseline), [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer), [aasist](https://github.com/clovaai/aasist) and [AIR-ASVspoof](https://github.com/yzyouzhang/AIR-ASVspoof).
