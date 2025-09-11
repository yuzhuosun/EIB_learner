# EIB_learner

## Overview

We provide the code of our paper "Understanding the Information Propagation Effects of Communication Topologies in LLM-based Multi-Agent Systems" EMNLP 2025 (main). Our code inherits the training and validation modules from G-designer, and the design of EIB_learner is placed in "G-designer/graph.py". (function arunfuse), and the experimental code is in `experiments` folder.

## Quick Start

### Install packages

pip install -r requirements.txt
```

### Add API keys in `template.env` and change its name to `.env`

```python
BASE_URL = "" # the BASE_URL of OpenAI LLM backend
API_KEY = "" # for OpenAI LLM backend
```

### Run GDesigner on MMLU by running the following scripts

```bash
python experiments/run_mmlu.py --mode FullConnected --batch_size 4 --agent_nums 6 --num_iterations 10 --num_rounds 1 --optimized_spatial
```


## Acknowledgement

This code refers to [G-designer](https://github.com/yanweiyue/GDesigner).
