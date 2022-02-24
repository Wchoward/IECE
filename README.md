# KJ-IECE
Knowledge-Enriched Joint-Learning Implicit Emotion Cause Extraction



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



