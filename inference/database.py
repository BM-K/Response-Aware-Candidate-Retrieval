import os
import csv
import torch
import faiss
import numpy as np

from utils import *
from tqdm import tqdm

class Database():

    def __init__(self, args):
        self.args = args
        self.file_list = os.listdir('./')
        self.h5file_name = f'{self.args.lang}_{self.args.retrieval_mode}_db_cache.hdf5'
        
    def offline_embedding(self,
                          retrieval_model=None,
                          tokenizer=None,
        ):
        print(f"\n\t===Start DB Embedding===")
        corpus = []
        corpus_sep = []
        corpus_data_path = {'train': self.args.corpus_train, 'valid': self.args.corpus_valid}
    
        for key, val in corpus_data_path.items():
            with open(corpus_data_path[key], "r", encoding="utf-8") as file:
                lines = csv.reader(file, delimiter="\t", quotechar='"')

                for line in lines:
                    query_prime, response_prime, _, _ = line
                    if self.args.retrieval_mode == 'base': corpus.append(query_prime + " [SEP] " + response_prime)
                    elif self.args.retrieval_mode == 'r_prime_sim': corpus.append(response_prime)
                    elif self.args.retrieval_mode == 'q_prime_sim': corpus.append(query_prime)
                    else:
                        raise Exception('===No Valid Mode===\n===Please Check args.retrieval_mode===')
                    corpus_sep.append(query_prime + " [SEP] " + response_prime)
    
        if self.h5file_name not in self.file_list:
            embedding_dim = self.args.embedding_dim
            get_db_batch = self.args.db_embedding_bsz

            tensor_stack = torch.ones(1, embedding_dim)
            for step, start in enumerate(tqdm(range(0, len(corpus), get_db_batch))):
                corpus_tokens = corpus[start: start + get_db_batch]

                with torch.no_grad():
                    corpus_tokens = tokenizer(corpus_tokens, padding=True, truncation=True, return_tensors="pt")
                    corpus_tokens.to(self.args.device)

                    corpus_embeddings, _ = retrieval_model(corpus_tokens)
                    corpus_embeddings = corpus_embeddings[:, :1].squeeze(1)

                tensor_stack = torch.cat([tensor_stack, corpus_embeddings.cpu()], dim=0)
                
            embedded_db = tensor_stack[1:]
            h5py_io(self.h5file_name, embedded_db, 'w')

        else:
            print(f'\t***LOAD CACHE DATABASE: {self.h5file_name}***')
            embedded_db = h5py_io(self.h5file_name, None, 'r')
        
        return embedded_db, corpus_sep, self.faiss_indexing(embedded_db.cpu().detach().numpy())

    def faiss_indexing(self,
                       embedded_db,
        ):
        print("\n\t===Training FAISS===")
        quantizer = faiss.IndexFlatL2(self.args.embedding_dim)
    
        index = faiss.IndexIVFFlat(quantizer, self.args.embedding_dim, self.args.num_centroids)
        index.train(embedded_db)
        index.add(embedded_db)
        
        print("\t===Complete FAISS===")
        
        return index


