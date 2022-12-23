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
    """
    value_list = []
    
    data_file_name = ['train.tsv']
    sorted_file_name = ['base_kb_new_train_only_kb.tsv']
    for cur_file in data_file_name:
        with open(cur_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.split('\t')
            
                query = line[0]
                if line[-1].split('|')[-1].strip() != 'None':
                    retrieved_value = line[-1].split('|')[:-1]
                    kb = line[-1].split('|')[-1].split('<sep>')
                    kb = [v.replace('"', '').replace('\n', '').strip() for v in kb]
                    
                    pattern = r'\([^)]*\)'    
                    kb = [re.sub(pattern=pattern, repl='', string = k) for k in kb]
                
                    tokenized_query = retrieval_tokenizer(query, truncation=True, return_tensors='pt',
                            max_length=args.max_len, padding='max_length')
                    tokenized_query.to(args.device)
                    
                    embedded_query, _ = retrieval_model(tokenized_query, response=True)
                    embedded_query = embedded_query[:, :1]
                    
                    tokenized_kb = retrieval_tokenizer(kb, truncation=True, return_tensors='pt',
                            max_length=args.max_len, padding='max_length')
                    tokenized_kb.to(args.device)

                    embedded_kb, _ = retrieval_model(tokenized_kb, response=True)
                    embedded_kb = embedded_kb[:, :1]
                    
                    sim = cos(embedded_query.squeeze(1), embedded_kb.squeeze(1))
                    
                    score, preds = sim.topk(k=1, dim=-1)
                    
                    if score>0.957:
                        #print(query)
                        #print(kb[preds])
                        #print(score)
                        #print("---------------")
                        check+=1
                        
                        preds = preds.squeeze(0) 
                
                    #logits = torch.matmul(embedded_query.squeeze(0), embedded_kb.squeeze(1).transpose(-1, -2))
                    #preds = logits.topk(k=1, dim=-1)[-1].squeeze(0)
                    
                        kb = kb[preds]
                    
                        retrieved_value = '|'.join(retrieved_value)
                        retrieved_value = f'{retrieved_value} | {kb}'
                        #value_list.append([query, line[1]])
                        value_list.append([query, line[1], retrieved_value])
                    
                    else:
                        kb = 'None'
                        retrieved_value = '|'.join(retrieved_value)
                        retrieved_value = f'{retrieved_value} | {kb}'
                #        value_list.append([query, line[1], retrieved_value])
                #else:
                #    kb = 'None'
                #    value_list.append([query, line[1], line[2].strip()])
                
    
    print(check)
    import csv
    with open(sorted_file_name[0], 'w', encoding='utf-8') as w:
        tw = csv.writer(w, delimiter='\t')
        for val in value_list:
            tw.writerow(val)
    exit()
    """
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
