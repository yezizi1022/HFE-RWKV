# HFE-RWKV
Official implementation code for [_HFE-RWKV: High-Frequency Enhanced RWKV  Model for Efficient Left Ventricle Segmentation in  Pediatric Echocardiograms_](https://ieeexplore.ieee.org/document/10888300) paper (Accepted by 2025 ICASSP as Oral!)

---

![Proposed Model](./images/proposed_method_v2.png)

---
## Prepare data

- The `DataSplit.py` script is used to divide the EchoNet-Pediatric dataset into three subsets.
- The `Prepare_US.py` script is designed to preprocess the dataset for input into the model. 

**Notice:**

**Dataset Name**: EchoNet-Pediatric  
**Project URL**: [https://echonet.github.io/pediatric/](https://echonet.github.io/pediatric/)

---
## Environment and Installation

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
---
## Train and Test
- The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train 

```bash
 python train.py --cfg configs/config_skin.yaml --path_to_data your DATA_DIR --saved_model your OUT_DIR
```

- Test 

```bash
python test.py
```
---
## Updates
- December 27, 2024: Accepted by ICASSP2025.
---
## References
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
---
## Citation
```
@INPROCEEDINGS{10888300,
  author={Ye, Zi and Chen, Tianxiang and Wang, Ziyang and Zhang, Hanwei and Zhang, Lijun},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={HFE-RWKV: High-Frequency Enhanced RWKV Model for Efficient Left Ventricle Segmentation in Pediatric Echocardiograms}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Accuracy;Codes;Shape;Computational modeling;Frequency-domain analysis;Medical services;Speech enhancement;Signal processing;Feature extraction;Computational complexity;Left ventricle segmentation;RWKV;frequency enhancement},
  doi={10.1109/ICASSP49660.2025.10888300}}

```
