import csv
import copy
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_sentence_embedding_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.sentence_embedder)
    model = AutoModel.from_pretrained(args.sentence_embedder)
    model.to(args.device)

    return model, tokenizer

def simple_statistics(data_stack):
    for key, value in data_stack.items(): print(key, len(value))

def load_data(data_path, data_stack):
    for step, cur_path in enumerate(data_path):
        split = ['train', 'test', 'dev']
        print(f"Load Dataset: {cur_path}")

        with open(cur_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                line = line.split('\t')
                query, response = line[0].strip(), line[1].strip()
                data_stack[split[step]].append((query, response))
    
    return True

def pairing_data(data_stack, corpus, pair):
    for key in pair.keys():
        for query, response in data_stack[key]:
            if key == 'train' or key == 'dev':
                corpus.append(query + " [SEP] " + response)
            
            pair[key].append(query + " [SEP] " + response)

def write_data(args, total_data, split, database, corpus_data_path):
    iter_ = 0
    for key, value in total_data.items():
        with open(split[iter_], "w", encoding="utf-8") as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            for line in value:
                if key != 'test':
                    query, response, search_idx, best_score_index = line
                    tsv_writer.writerow([query, response, search_idx, best_score_index])
                else:
                    query, response, search_idx, best_score_index, labels = line
                    tsv_writer.writerow([query, response, search_idx, best_score_index, labels])
        iter_ += 1

    iter_ = 0
    for key, value in database.items():
        if key != 'test':
            with open(corpus_data_path[iter_], "w", encoding="utf-8") as file:
                tsv_writer = csv.writer(file, delimiter='\t')
                for line in value:
                    current_response = line[:args.top_k-1]
                    search_idx, query = line[args.top_k-1:]

                    current_response = [r for r, score in current_response]
                    current_response.append(search_idx)
                    current_response.append(query)

                    tsv_writer.writerow(current_response)
        iter_ += 1

def embedding_database(args, corpus, model, tokenizer, mode):
    print(f"Start {mode} Embedding \n")
    tensor_stack = torch.ones(1, 768)
    for step, start in enumerate(tqdm(range(0, len(corpus), args.embedding_batch_size))):
        corpus_tokens = corpus[start: start + args.embedding_batch_size]

        with torch.no_grad():
            corpus_tokens = tokenizer(corpus_tokens, padding=True, truncation=True, return_tensors="pt")
            corpus_tokens.to(args.device)
            corpus_embeddings = model(**corpus_tokens, output_hidden_states=True, return_dict=True).pooler_output

        tensor_stack = torch.cat([tensor_stack, corpus_embeddings.cpu()], dim=0)
    
    return tensor_stack
    
def labeling(args, corpus, stack, mode='None'):
    print(f"Start {mode} set Labeling \n")
    start, end = 0, 0
    tensor_stack = torch.ones(1, 1, dtype=torch.int)

    for step, start in enumerate(tqdm(range(0, len(stack), args.embedding_batch_size))):
        end += args.embedding_batch_size
        cos_scores = util.pytorch_cos_sim(stack[start:end].to(args.device), corpus.to(args.device))[:]
        cos_scores = cos_scores.cpu()
        
        if mode == 'train' or mode == 'dev':
            _results = np.argpartition(-cos_scores, range(args.top_k))[:, 1:args.top_k]
        else:
            _results = np.argpartition(-cos_scores, range(args.top_k))[:, 0:1]

        tensor_stack = torch.cat([tensor_stack, _results], dim=0)
        start += args.embedding_batch_size

    results = tensor_stack[1:]
    
    return results

def indexing(cur_idx, corpus, pair, results, database, total_data, mode='none'):
    search_idx = cur_idx
    for idx, values in enumerate(pair[mode]):
        q, r = values.split('[SEP]')
        q = q.strip()
        r = r.strip()
    
        top_k_setting = [(corpus[results[idx]], 1), search_idx, q]
    
        database[mode].append(top_k_setting)
        
        if mode=='test':
            total_data[mode].append((q, r, search_idx, 0, (int(results[idx]))))
        else:
            total_data[mode].append((q, r, search_idx, 0))
    
        search_idx += 1

def main(args):
    set_seed(args.seed)
    model, tokenizer = get_sentence_embedding_model(args)
    
    data_path = [args.train_data, args.test_data, args.dev_data]
    data_stack = {'train': [], 'test': [], 'dev': []}

    completion = load_data(data_path, data_stack)
    #data_stack['train'] = data_stack['train'][:10000]
    #data_stack['test'] = data_stack['test'][:1000]
    #data_stack['dev'] = data_stack['dev'][:1000]
    assert completion

    train_length = len(data_stack['train'])
    dev_length = len(data_stack['dev'])
    simple_statistics(data_stack)
    
    corpus, pair = [], {'train': [], 'test': [], 'dev': []}
    pairing_data(data_stack, corpus, pair)
    length_of_corpus = len(corpus)
    
    tensor_stack = embedding_database(args, corpus, model, tokenizer, mode='Database')
    corpus_stack = tensor_stack[1:]
    print(f"Total DB embedding length: {len(corpus_stack)}")
    
    train_stack = copy.deepcopy(corpus_stack[:train_length])
    print(f'train stack: {len(train_stack)}')

    tensor_stack = embedding_database(args, pair['dev'], model, tokenizer, mode='Dev Set')
    dev_stack = tensor_stack[1:]
    print(f'dev stack: {len(dev_stack)}')
    
    tensor_stack = embedding_database(args, pair['test'], model, tokenizer, mode='Test Set')
    test_stack = tensor_stack[1:]
    print(f'test stack: {len(test_stack)}')
    
    train_results = labeling(args, corpus_stack, train_stack, mode='train')
    dev_results = labeling(args, corpus_stack, dev_stack, mode='dev')
    test_results = labeling(args, corpus_stack, test_stack, mode='test')

    print(train_results[:10])
    print(dev_results[:10])
    print(test_results[:10])
    
    database = {'train': [], 'test': [], 'dev': []}
    total_data = {'train': [], 'test': [], 'dev': []}
    
    indexing(0, corpus, pair, train_results, database, total_data, mode='train')

    database['dev'] = copy.deepcopy(database['train'])
    indexing(len(pair['train']), corpus, pair, dev_results, database, total_data, mode='dev')

    indexing(0, corpus, pair, test_results, database, total_data, mode='test')

    split = [args.labeled_train_data, args.labeled_test_data, args.labeled_dev_data]
    corpus_data_path = [args.train_db, args.test_db, args.dev_db]
    
    write_data(args, total_data, split, database, corpus_data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="dataset/train.tsv")
    parser.add_argument("--dev_data", type=str, default="dataset/dev.tsv")
    parser.add_argument("--test_data", type=str, default="dataset/test.tsv")
    
    parser.add_argument("--labeled_train_data", type=str, default="RACAR/data/labeled_train.tsv")
    parser.add_argument("--labeled_dev_data", type=str, default="RACAR/data/labeled_dev.tsv")
    parser.add_argument("--labeled_test_data", type=str, default="RACAR/data/labeled_test.tsv")
    
    parser.add_argument("--train_db", type=str, default="RACAR/data/train_db.tsv")
    parser.add_argument("--dev_db", type=str, default="RACAR/data/dev_db.tsv")
    parser.add_argument("--test_db", type=str, default="RACAR/data/test_db.tsv")

    parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument("--sentence_embedder", type=str, default="BM-K/KoSimCSE-roberta")
    parser.add_argument("--embedding_batch_size", type=int, default=512)    
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--hard_label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    
    args = parser.parse_args()

    main(args)
