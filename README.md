# A Hybrid Response Generation by Response-Aware Candidate Retrieval and Seq-to-seq Generation
This repository contains the code for our paper "A Hybrid Response Generation by Response-Aware Candidate Retrieval and Seq-to-seq Generation" [@IP&M (SCIE)](https://www.sciencedirect.com/journal/information-processing-and-management) (IF: 7.466).

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
RACAR retrieves the most similar candidate from Database (DB) using both a user query and its golden response. The fact that it utilizes both query and golden response is a distinguishable merit of RACAR from the existing candidate retrieval models. Actually, RACAR is a simple classifier which consists of a transformer encoder layer and a similarity softmax layer. This paper adopts BERT [(Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) as the transformer encoder.
After training RACAR, construct data for training REGE. The labeled data sets for training REGE are automatically moved to `REGE/data/` directory. 
> **Note** <br>
> `--embedding_mask` is number of your training data
```
cd RACAR
bash run_racar.sh
```

## Training REGE
Following the decoder fusion in the work of [Izacard and Grave (2021)](https://arxiv.org/abs/2007.01282), the *K* candidate answers are respectively concatenated to query. This paper adopts BART [(Lewis et al., 2020)](https://arxiv.org/abs/1910.13461) as the encoder-decoder model. Therefore, training REGE is equivalent to fine-tuning the BART.
```
cd REGE
bash run_rege.sh
```

## Inference
Note that the comparisons of (query, prospective response) with all (candidate question, candidate answer) ∈ DB could take a very long time since DB is usually extremely large. Therefore, FAISS [(Jonhonson et al., 2017)](https://arxiv.org/abs/1702.08734) is adopted for the speedup of this comparison process.
```
cd inference
bash run_eval.sh
```

## Demo with FastAPI
<img src = "https://user-images.githubusercontent.com/55969260/200460525-ac04b760-0b66-4371-84f5-d82f15d1b1e6.gif"> <br>
