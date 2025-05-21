# DeSocial: Blockchain-based Decentralized Social Networks

This repository contains the source code for the DeSocial project. DeSocial is a decentralized social network framework that utilize blockchain infrastructure and graph learning models to enable transparent, user-driven, and personalized social network predictions.

## 🌐 Project Structure

```
DeSocial/
│
├── blockchain/ # Blockchain simulation environment and client scripts
├── contract/ # Solidity smart contracts for validator voting, user actions, etc.
├── model/ # Graph learning models, and personalized algorithm selection module (e.g., GCN, GAT, etc.)
├── utils/ # Utility functions and helpers
├── eval.py # Evaluation functions
└── run.py # Main entry point to run the pipeline
```

## 🏗️ Framework

Here gives the framework of DeSocial (both modules enabled).

![DeSocial](img/DeSocial.png)

**Step 1:** User $p_i$ submits requests to predict social links with target nodes $q_{1_{p_i}}, q_{2_{p_i}}, \dots$.
    
**Step 2:** The blockchain collects all user requests, constructs $\mathcal{G}^{t+1}$, and assigns a validator community $\Phi$ according to each backbone model $\mathcal{F}_i\in \mathcal{F}$ through the smart contract.
    
**Step 3:** Each validator $\phi \in \mathcal{V}_{val}^t$ independently trains their own graph learning model $f_{\theta_\phi}$ based on the data $\mathcal{D}^t$ stored in their own local memory. $\mathcal{D}^t$ describes the union of the historical snapshots $\mathcal{G}^0, \mathcal{G}^1, ..., \mathcal{G}^t$, and each node stored one copy of $\mathcal{D}^t$.
    
**Step 4:** User $p_i$ creates a personalized neighborhood sampling task based on local graph structure.
    
**Step 5:** Validator nodes retrieve $p_i$'s request  through the blockchain smart contract, and evaluate it using different available algorithms $\mathcal{F}_j$.
    
**Step 6:** One selected validator in each community executes the sampling task using algorithm $\mathcal{F}_j$ and returns results to the blockchain through the smart contract.
    
**Step 7:** The result of each algorithm trial is returned to $p_i$ through the blockchain for evaluation.
    
**Step 8:** User $p_i$ selects a preferred model $\mathcal{F}_{p_i}$ based on the returned results.
    
**Step 9:** Validators in $\Phi$ run $\mathcal{F}_{p_i}$ on $p_i$'s request and submit their binary votes to the blockchain. The blockchain aggregates the votes to form the final prediction $\mathcal{G}^{t+1}_{pred}$. Both the voting and aggregating operations are defined by the smart contract.

**Step 10:** The period ends, all the nodes in the network updates their local social network data $\mathcal{D}^t$ via requesting the blockchain for the latest links by the smart contract.

For the details of these notations, please refer the problem definitions in our paper.

## ✈️ Quickstart

After downloading the repo to your computer/server, please install all the dependencies by:
```bash
pip install -r requirements.txt
```

For the setup of ETH Ganache environment, please follow `ganache_install.md`.

To run DeSocial, please use
```bash
python run.py
```

To quickly reproduce the result of DeSocial in the best configuration, please use
```bash
python run.py --cuda $CUDA --dataset_name $DATASET --f_pool $F --experts $EXPERTS --metric $METRIC --load_best_configs
```

```
$F in [MLP, GCN, GAT, SAGE, SGC]
$F 
```

For example, if you want to reproduce DeSocial-X (with validator community size of 5), X is one of the backbones, let's say SGC on UCI, please use
```bash
python run.py --cuda 0 --dataset_name UCI --f_pool SGC --experts 5 --load_best_configs
```

If you want to reproduce DeSocial-PA on UCI, please use
```bash
python run.py --cuda 0 --dataset_name UCI --f_pool PA --load_best_configs
```

If you want to reproduce DeSocial-Full on UCI, please use
```bash
python run.py --cuda 0 --dataset_name UCI --f_pool PA --experts 5 --load_best_configs
```

If you want to reproduce DeSocial on UCI at a given backbone selection pool {GraphSAGE, SGC}, please use
```bash
python run.py --cuda 0 --dataset_name UCI --f_pool SAGE+SGC --experts 5 --load_best_configs
```
use "+" to combine the backbone names.