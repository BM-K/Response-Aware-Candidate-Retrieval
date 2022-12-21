# Response-Aware-Candidate-Retrieval
This repository contains the code for our paper "A Hybrid Response Generation by Response-Aware Candidate Retrieval and Seq-to-seq Generation" [@IP&M](https://www.sciencedirect.com/journal/information-processing-and-management) (IF: 7.466).

## Overview
<img src='https://user-images.githubusercontent.com/55969260/208830103-d8acfa42-3d4b-444e-8a75-8fc81cc4200a.png'> <br>
The previous studies on the retrieval-based models tried to find the most relevant candidate pairs by matching the user query with a candidate question or a candidate answer, which leads to a poor retrieval performance due to a lack of information within a single source. As a solution this problem, this paper proposes a novel hybrid response generator which consists of **R**esponse-**A**ware **CA**ndidate **R**etriever (**RACAR**) and **RE**sponse **GE**nerator (**REGE**). RACAR searches for relevant candidate pairs using both a user query and a golden response to the query, and REGE, a sequence-to-sequence generator, outputs a final response using the user query and the candidate answers of candidate pairs retrieved by RACAR. Since RACAR uses both a user query and its golden response to retrieve relevant candidate pairs from the database, it outperforms other retrieval models that use any single source.

## Setups
[![Python](https://img.shields.io/badge/python-3.8.5-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers|4.24.0-pink?color=FF33CC)](https://github.com/huggingface/transformers)

- Install packages
```
pip install -r requirements.txt
```
