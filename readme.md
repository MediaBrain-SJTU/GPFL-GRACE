#  GRACE: Enhancing Federated Learning for Medical Imaging with Generalized and Personalized Gradient Correction - MICCAI 2023

### Paper of our work

This repo provides a demo for the MICCAI 2023 paper "GRACE: Enhancing Federated Learning for Medical Imaging with Generalized and Personalized Gradient Correction".

paper link: (wait for camera ready version)

To cite, please use:

```latex

```


### Structure of our code

Unfinished (code for GRACE and TTA part)!

```shell
├── algorithms
│   ├── __init__.py
│   ├── ditto.py
│   ├── elcfs.py
│   ├── fed_distance.py
│   ├── fedavg.py
│   ├── fedbabu.py
│   ├── fedbn.py
│   ├── fedmtl.py
│   ├── fedper.py
│   ├── fedprox.py
│   ├── fedrep.py
│   ├── fedrod.py
│   ├── grace_client.py
│   ├── grace_fl.py
│   ├── grace_server.py
│   ├── harmo_fl.py
│   ├── meta_trainer.py
│   ├── moon.py
│   ├── perfedavg.py
│   ├── perfedme.py
│   └── scaffold.py
├── configs
│   └── default.py
├── data
│   ├── Fourier_Aug.py
│   ├── __init__.py
│   ├── a_distance.py
│   ├── flamby_fed_isic2019.py
│   ├── isic2019_dataset.py
│   ├── meta_dataset.py
│   ├── metadata
│   │   └── isic2019_train_test_split
│   └── prostate_dataset.py
├── networks
│   ├── FedOptimizer
│   │   ├── FedProx.py
│   │   ├── HarmoFL.py
│   │   ├── PerFedAvg.py
│   │   ├── PerFedMe.py
│   │   ├── Scaffold.py
│   │   └── __pycache__
│   ├── GRL.py
│   ├── __init__.py
│   ├── amp_utils.c
│   ├── get_network.py
│   ├── isic_model.py
│   ├── prostate_model.py
│   └── setup.py
├── readme.md
├── runs
│   └── run_trainer.py
├── utils
│   ├── classification_metric.py
│   ├── log_utils.py
│   ├── segmentation_metric.py
│   └── test_a_distance.py
└── visualization

```


### Requirements

- Python 3.9.7
- numpy 1.20.3
- torch 1.11.0
- torchvision 0.12.0

### Datasets

- Fed-Prostate (Prostate dataset in [FedDG-ELCFS](https://github.com/liuquande/FedDG-ELCFS) and [SAML](https://liuquande.github.io/SAML/))
- Fed-ISIC (Fed-ISIC2019 in [Flamby](https://github.com/owkin/FLamby))


### Training code
```python
python ./runs/run_trainer.py --algorithm FedAvg_Prostate_Trainer --dataset prostate --model prostate_unet --align_weight 0.1 --align_warmup 0 --align_type CORAL --batch_size 16 --lr 1e-3 --optimizer adam --lr_policy step --local_epochs 1 --rounds 200 --note baseline_prostate

python ./runs/run_trainer.py --algorithm FedAvg_ISIC_Trainer --dataset isic --model isic_b0 --align_weight 0.1 --align_warmup 0 --align_type CORAL --batch_size 64 --lr 5e-4 --optimizer adam --lr_policy step --local_epochs 5 --rounds 40 --note baseline_isic
```


### Acknowledgement

Part of our code is borrowed from the following repositories.

- FedDG-GA [https://github.com/MediaBrain-SJTU/FedDG-GA]
- Flamby [https://github.com/owkin/FLamby]
- FACT [https://github.com/MediaBrain-SJTU/FACT]
- FedDG-ELCFS [https://github.com/liuquande/FedDG-ELCFS]
- SCAFFOLD-PyTorch [https://github.com/KarhouTam/SCAFFOLD-PyTorch]
  
We thank to the authors for releasing their codes. Please also consider citing their works.

