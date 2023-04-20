from database import Database
from setting import Arguments, Setting
from generation import DialogueGeneration

import torch
import torch.nn as nn
from tqdm import tqdm
import re

def main(args):
    cos = nn.CosineSimilarity(dim=-1)
    check = 0

    retrieval_model, generative_model, retrieval_tokenizer, generative_tokenizer = Setting(args).model()
    embedded_database, corpus, faiss_index = Database(args).offline_embedding(retrieval_model, retrieval_tokenizer)
    
    dg = DialogueGeneration(args=args,
                            retrieval_model=retrieval_model,
                            generative_model=generative_model,
                            retrieval_tokenizer=retrieval_tokenizer,
                            generative_tokenizer=generative_tokenizer,
                            embedded_database=embedded_database,
                            corpus=corpus,
                            faiss_index=faiss_index,)

    if args.eval == 'True':
        dg.eval_inference()
    else:
        dg.user_inference()

if __name__ == '__main__':
    args = Arguments().run()
    main(args)
