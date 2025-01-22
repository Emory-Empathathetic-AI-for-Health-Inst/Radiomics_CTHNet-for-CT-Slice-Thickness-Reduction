# Download model checkpoint here: https://emory-my.sharepoint.com/:u:/g/personal/pmutha_emory_edu/EfUOXrzk7q5HgQ2FX8aEqA0BAQt9sUCjw8UOGt2gFiZzAA?e=AeoV8S
Thanks to Zelin for downloading the weights. Please contact Pushkar Mutha (pmutha@emory.edu) if you need additional details or help setting this up.

# Spatial resolution enhancement using deep learning improves chest disease diagnosis based on thick slice CT (*npj Digital Medicine*)
[Pengxin Yu](https://github.com/smilenaxx/), [Haoyue Zhang](https://github.com/zhanghaoyue), Dawei Wang, Rongguo Zhang, et al.

[![paper](https://img.shields.io/badge/npj_Digital_Medicine-Paper-green)](https://www.nature.com/articles/s41746-024-01338-8)

## Abstract
> CT is crucial for diagnosing chest diseases, with image quality affected by spatial resolution. Thick-slice CT remains prevalent in practice due to cost considerations, yet its coarse spatial resolution may hinder accurate diagnoses. Our multicenter study develops a deep learning synthetic model with Convolutional-Transformer hybrid encoder-decoder architecture for generating thin-slice CT from thick-slice CT on a single center (1576 participants) and access the synthetic CT on three cross-regional centers (1228 participants). The qualitative image quality of synthetic and real thin-slice CT is comparable (p= 0.16). Four radiologists’ accuracy in diagnosing community-acquired pneumonia using synthetic thin-slice CT surpasses thick-slice CT (p < 0.05), and matches real thin-slice CT (p > 0.99). For lung nodule detection, sensitivity with thin-slice CT outperforms thick-slice CT (p < 0.001) and comparable to real thin-slice CT (p > 0.05). These ndings indicate the potential of our model to generate high-quality synthetic thin-slice CT as a practical alternative when real thin-slice CT is preferred but unavailable.

## Network Architecture
<img width="1063" alt="截屏2024-12-04 11 16 38" src="https://github.com/user-attachments/assets/4a3f1149-fc7f-4408-ae89-4733ac8a9343">

## Visual Comparison Results
<img width="1242" alt="截屏2024-12-04 11 17 41" src="https://github.com/user-attachments/assets/838a71ba-8e55-4a07-87d1-4061fff1a3b9">


## Code
**Model train**
```
python train.py train --path_key HD --gpu_idx 0 --model t3dv1 --net_idx CTH_net
```
**Model val**
```
python val.py val --path_key HD --gpu_idx 0 --model t3dv1 --net_idx CTH_net
```
**Model test**
```
python test.py test --path_key HD --gpu_idx 0 --model t3dv1 --net_idx CTH_net
```

## Model
Link: https://emory-my.sharepoint.com/:u:/g/personal/pmutha_emory_edu/EfUOXrzk7q5HgQ2FX8aEqA0BAQt9sUCjw8UOGt2gFiZzAA?e=AeoV8S
The model path is **/model/xxx.pkl**.
Thanks to Zelin Zhang for downloading the weights from Baidu. Please contact Pushkar Mutha (pmutha@emory.edu) if you need additional details or help setting this up.
~~The well-trained model parameters can be downloaded in [baidu cloud disk](https://pan.baidu.com/s/1Z245Q9NzUjg8bZjkj1YWtA?pwd=tg55).~~


## Data
The data used for this study are not publicly available due to hospital privacy restrictions.  
In our previous [paper](https://github.com/smilenaxx/RPLHR-CT/), we made public a dataset **RPLHR-CT**, which contains 250 cases of data.   
We build a tiny dataset with 40 cases to [baidu cloud disk](https://pan.baidu.com/s/1QXbcFuWAHOiY3FijEcsowQ?pwd=ut7p).  
For data usage rights reasons, should you required the complete dataset, please contact **ypengxin@infervision.com**.

For this code, the data should be organized as follows.
```
data/
|-- HD_1mm/
|  |-- A.nii.gz
|  |-- B.nii.gz
|  |-- C.nii.gz
|-- HD_5mm/
|  |-- A.nii.gz
|  |-- B.nii.gz
|  |-- C.nii.gz
```

## Citation
If you use our code or data, please consider citing:
```
@article{yu2024spatial,
  title={Spatial resolution enhancement using deep learning improves chest disease diagnosis based on thick slice CT},
  author={Yu, Pengxin and Zhang, Haoyue and Wang, Dawei and Zhang, Rongguo and Deng, Mei and Yang, Haoyu and Wu, Lijun and Liu, Xiaoxu and Oh, Andrea S and Abtin, Fereidoun G and others},
  journal={npj Digital Medicine},
  volume={7},
  number={1},
  pages={335},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
@inproceedings{yu2022rplhr,
  title={RPLHR-CT Dataset and Transformer Baseline for Volumetric Super-Resolution from CT Scans},
  author={Yu, Pengxin and Zhang, Haoyue and Kang, Han and Tang, Wen and Arnold, Corey W and Zhang, Rongguo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={344--353},
  year={2022},
  organization={Springer}
}
```

**Acknowledgment**: This code is based on the [RPLHR-CT Dataset and Transformer Baseline for Volumetric Super-Resolution from CT Scans](https://github.com/smilenaxx/RPLHR-CT/).


