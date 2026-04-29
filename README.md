# fraud-fusion Framework
Code and configuration for the paper 'A Temporal Hypergraph Transformer Fusion Framework with Fraud Ring Topology Analysis and Budget Aware Investigation for Credit Card Fraud Detection'

Author identifying information has been removed for review.

## Overview

A two-stage fraud detection framework with:
- Stage 1: A trust-gated hypergraph + heterogeneous graph
  transformer fusion model for transaction-level scoring.
- Stage 2: A temporally-causal ring extractor and budget-aware
  adaptive coupling alpha*(K) for analyst-facing prioritization.

Evaluated on the IEEE-CIS Fraud Detection benchmark.

## Requirements

- Python 3.10+
- PyTorch 2.1+
- PyTorch Geometric 2.5+
- pandas, numpy, scikit-learn, networkx, xgboost, scipy
- Single GPU with at least 16 GB memory (tested on H100 80 GB)

Install dependencies:

    pip install -r requirements.txt

## Dataset

Download the IEEE-CIS Fraud Detection dataset from
https://www.kaggle.com/c/ieee-fraud-detection/data and place
`train_transaction.csv` and `train_identity.csv` in `data/raw/`.

## Reproducing the experiments

Experiment 1 (detection at scale, N = Full Dataset):

    python src/main.py --config configs/experiment1.yaml --seed 42

Experiment 2 (ring stage, N = 45,000):

    python src/main.py --config configs/experiment2.yaml --seed 42

Output CSVs are written to `results/`. Wall-clock time on a
single H100 is approximately 1 hours for Experiment 1 and 3 hours approx
for Experiment 2 based on GPU memory capacity.

## Repository structure

    .
    ├── README.md
    ├── requirements.txt
    ├── LICENSE
    ├── configs/                    # YAML hyperparameter files
    ├── src/
    │   ├── main.py                 # entry point
    │   ├── models/                 # Hypergraph, HGT, Fusion
    │   ├── data/                   # dataset loaders, time splits
    │   └── ring_stage/             # causal extractor, alpha* optimizer
    └── results/                    # output CSVs from published runs

## License

MIT License. See LICENSE.
