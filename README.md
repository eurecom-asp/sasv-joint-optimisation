# sasv-joint-optimisation 

This repository provides codes to reproduce our paper [On the potential of jointly-optimised solutions to spoofing attack detection and automatic speaker verification](https://arxiv.org/pdf/2209.00506.pdf) accepted to the IberSPEECH 2022 conference.

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
python main.py --config ./config/ResNetSE34V2_AASIST_OC.conf
```
To train a back-end classifier and keep ASV and CM fixed:
```
python main.py --config ./config/ResNetSE34V2_AASIST_OC_fixed.conf
```

### Pre-trained models
We provide pre-trained models in `'pre_trained_models/'`.

You can try to calculate the evaluation score by:
```
python save_sasv_score.py
```

### References
If you find this repository useful, please consider citing:
```
@inproceedings{sasv_joint,
  author={Wanying Ge and Hemlata Tak and Massimiliano Todisco and Nicholas Evans},
  title={{On the potential of jointly-optimised solutions to spoofing attack detection and automatic speaker verification}},
  year=2022,
  booktitle={Proc. IberSPEECH 2022 (to appear)},
}
```


### Acknowledgements
Codes are based on the implementations of [SASVC2022_Baseline](https://github.com/sasv-challenge/SASVC2022_Baseline), [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer), [aasist](https://github.com/clovaai/aasist) and [AIR-ASVspoof](https://github.com/yzyouzhang/AIR-ASVspoof).
