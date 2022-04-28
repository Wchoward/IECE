# KJ-IECE
This repository contains the code for our CAAI Transactions on Intelligence Technology paper:

Chenghao Wu, Shumin Shi, Jiaxing Hu and Heyan Huang. Knowledge‐enriched joint‐learning model for implicit emotion cause extraction. CAAI Trans. Intell. Technol. 1–11(2022) [pdf](http://doi.org/10.1049/cit2.12099)

Please cite our paper if you use this code.



## Requirements

- Python3
- Tensorflow 1.14.0



## Overview

- `data/` contains the data used in this study
  - Due to the policy of *Beijing Engineering Research Center of High Volume Language Information Processing and Cl*, we are not able to release all the data in a public repo. Please send email to ` bjssm@bit.edu.cn`  to request full data.
- `src/` contains the scripts of  KJ-IECE model.
- `src/` contains the trained model files of  KJ-IECE.
- `docs/` contains the appendix of IECE paper.



## Usage

- KJ-IECE ($\lambda = 0.25$)

```bash
python src/train_k_fold.py --integrate_ek=True --lam=0.25 
```

- Introduce External Knowledge with Different $\lambda$

```bash
python src/train_k_fold.py --integrate_ek=True --lam=1.0
python src/train_k_fold.py --integrate_ek=True --lam=0.75
python src/train_k_fold.py --integrate_ek=True --lam=0.5
python src/train_k_fold.py --integrate_ek=True --lam=0.0
```

- Not introduce External Knowledge ($None$)

```bash
python src/train_k_fold.py --integrate_ek=False
```


## Citation

Please consider citing the following paper when using our code or pretrained models for your application.

```
@article{https://doi.org/10.1049/cit2.12099,
author = {Wu, Chenghao and Shi, Shumin and Hu, Jiaxing and Huang, Heyan},
title = {Knowledge-enriched joint-learning model for implicit emotion cause extraction},
journal = {CAAI Transactions on Intelligence Technology},
keywords = {emotion cause extraction, external knowledge fusion, implicit emotion recognition, joint learning},
doi = {https://doi.org/10.1049/cit2.12099},
url = {https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/cit2.12099},
eprint = {https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/cit2.12099},
}

```


