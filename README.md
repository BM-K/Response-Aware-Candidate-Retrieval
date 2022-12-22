# Response-Aware-Candidate-Retrieval
This repository contains the code for our paper "A Hybrid Response Generation by Response-Aware Candidate Retrieval and Seq-to-seq Generation" [@IP&M](https://www.sciencedirect.com/journal/information-processing-and-management) (IF: 7.466).

## Overview
<img src='https://user-images.githubusercontent.com/55969260/208833618-034cfad2-03f4-4387-875a-db0cd0b23fcc.png'> <br>
The previous studies on the retrieval-based models tried to find the most relevant candidate pairs by matching the user query with a candidate question or a candidate answer, which leads to a poor retrieval performance due to a lack of information within a single source. As a solution this problem, this paper proposes a novel hybrid response generator which consists of **R**esponse-**A**ware **CA**ndidate **R**etriever (**RACAR**) and **RE**sponse **GE**nerator (**REGE**). RACAR searches for relevant candidate pairs using both a user query and a golden response to the query, and REGE, a sequence-to-sequence generator, outputs a final response using the user query and the candidate answers of candidate pairs retrieved by RACAR. Since RACAR uses both a user query and its golden response to retrieve relevant candidate pairs from the database, it outperforms other retrieval models that use any single source.

## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers|4.24.0-pink?color=FF33CC)](https://github.com/huggingface/transformers)

- Install packages
```
pip install -r requirements.txt
```
## Construction of Training Data for RACAR
In order to train **RACAR**, it must be known in advance which candidate pair in database is most relevant to every (user query, golden response) pair. After labeling is complete, the labeled data sets are automatically moved to `RACAR/data/` directory.
> **Warning** <br>
> Since this repository provides only part of the AI-Hub data, individuals have to download the data set for complete training

```
bash run_labeling.sh
```

## Training RACAR
After training RACAR, construct data for training REGE. The labeled data sets for training REGE are automatically moved to `REGE/data/` directory. 
> **Note** <br>
> `--embedding_mask` is number of your training data
```
bash run_racar.sh
```

## Training REGE
```
bash run_rege.sh
```

## API Inference
```
python main.py \
  --lang ko \
  --model hybrid \
  --corpus_v1 data/database_v1.tsv \
  --corpus_v2 data/database_v2.tsv \
  --retrieval_ckpt models/outputs/single_retrieval_model.pt \
  --generative_ckpt models/outputs/single_generative_model.pt \
  --retrieval_mode base \
  --num_centroids 64 \
  --n_beams 5 \
  --min_length 3 \
  --db_embedding_bsz 256 \
  --num_ks 3 \
  --max_len 80 
```

## Demo with FastAPI
<img src = "https://user-images.githubusercontent.com/55969260/200460525-ac04b760-0b66-4371-84f5-d82f15d1b1e6.gif"> <br>
