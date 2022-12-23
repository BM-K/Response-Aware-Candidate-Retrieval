import csv
import torch
import logging

from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

import transformers
transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, mode='normal'):
        self.args = args
        self.mode = mode
        self.metric = metric

        self.tokenizer = tokenizer
        self.file_path = file_path

        self.labels = []
        self.input_ids = []
        self.attention_mask = []
        self.token_type_ids = []
        self.search_idx = []

        self.q_input_ids = []
        self.q_attention_mask = []
        self.q_token_type_ids = []

        """
        BERT
        [CLS] 101
        [PAD] 0
        [UNK] 100
        """
        self.init_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)

    def load_data(self, type):
        st = 0
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')

            # Data pre-processing
            logger.info('Data pre-processing')
            for step, line in enumerate(tqdm(lines)):
                self.data2tensor(line)
                #st += 1
                #if st == 4000:
                #    break
        assert len(self.input_ids) == \
               len(self.attention_mask) == \
               len(self.labels)

    def data2tensor(self, line):

        if self.args.retrieval_mode == 'base':
            if self.mode == 'normal':
                question, response, search_idx, label = line[0].strip(), \
                                                        line[1].strip(), \
                                                        line[2].strip(), \
                                                        line[3].strip()
            else:
                if self.args.test_data.find('test') != -1:
                    question, response, search_idx, _, _ = line[0].strip(), \
                                                           line[1].strip(), \
                                                           line[2].strip(), \
                                                           line[3].strip(), \
                                                           line[4].strip()
                    label = 0
                else:
                    question, response, search_idx, label = line[0].strip(), \
                                                            line[1].strip(), \
                                                            line[2].strip(), \
                                                            line[3].strip()
            sentence_tokens = self.tokenizer(question,
                                             response,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             padding='max_length')

            only_question = self.tokenizer(question,
                                           truncation=True,
                                           return_tensors="pt",
                                           max_length=self.args.max_len,
                                           padding='max_length')

            self.q_input_ids.append(only_question['input_ids'].squeeze(0))
            self.q_attention_mask.append(only_question['attention_mask'].squeeze(0))
            self.q_token_type_ids.append(only_question['token_type_ids'].squeeze(0))

        elif self.args.retrieval_mode == 'q_prime_sim' or self.args.retrieval_mode == 'r_prime_sim':
            if self.mode == 'normal':
                question, response, search_idx, label = line[0].strip(), \
                                                        line[1].strip(), \
                                                        line[2].strip(), \
                                                        line[3].strip()
            else:
                if self.args.test_data.find('test') != -1:
                    question, response, search_idx, _, _ = line[0].strip(), \
                                                           line[1].strip(), \
                                                           line[2].strip(), \
                                                           line[3].strip(), \
                                                           line[4].strip()
                    label = 0
                else:
                    question, response, search_idx, label = line[0].strip(), \
                                                            line[1].strip(), \
                                                            line[2].strip(), \
                                                            line[3].strip()

            sentence_tokens = self.tokenizer(question,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             padding='max_length')
        else:
            print("==Retrieval Mode ERROR (Tokenizing)==")
            sentence_tokens, search_idx, label = None
            exit()

        self.input_ids.append(sentence_tokens['input_ids'].squeeze(0))
        self.attention_mask.append(sentence_tokens['attention_mask'].squeeze(0))
        self.token_type_ids.append(sentence_tokens['token_type_ids'].squeeze(0))
        self.search_idx.append(int(search_idx))
        self.labels.append(int(label))

        return True

    def __getitem__(self, index):
        if self.args.retrieval_mode == 'base':
            input_data = {'input_ids': self.input_ids[index].to(self.args.device),
                          'attention_mask': self.attention_mask[index].to(self.args.device),
                          'token_type_ids': self.token_type_ids[index].to(self.args.device),
                          'q_input_ids': self.q_input_ids[index].to(self.args.device),
                          'q_attention_mask': self.q_attention_mask[index].to(self.args.device),
                          'q_token_type_ids': self.q_token_type_ids[index].to(self.args.device),
                          'search_idx': torch.IntTensor([self.search_idx[index]]).to(self.args.device),
                          'labels': torch.LongTensor([self.labels[index]]).to(self.args.device)}

        else:
            input_data = {'input_ids': self.input_ids[index].to(self.args.device),
                          'attention_mask': self.attention_mask[index].to(self.args.device),
                          'token_type_ids': self.token_type_ids[index].to(self.args.device),
                          'search_idx': torch.IntTensor([self.search_idx[index]]).to(self.args.device),
                          'labels': torch.LongTensor([self.labels[index]]).to(self.args.device)}

        return input_data

    def __len__(self):
        return len(self.labels)


def get_loader(args, metric):
    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.test_data

    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer)
        valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer)
        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer, mode='test')
        test_iter.load_data('test')

        # Test time must set shuffle=False
        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=False)}

    else:
        logger.info("Error: None type loader")
        exit()

    return loader, tokenizer


def test_labels(args):
    total_test_labels_list = []
    path_to_test_db_data = args.path_to_data + '/' + args.test_data
    with open(path_to_test_db_data, "r", encoding="utf-8") as file:
        lines = csv.reader(file, delimiter="\t", quotechar='"')
        for line in lines:
            labels = line[-1]
            total_test_labels_list.append([int(labels)])

    return total_test_labels_list


def post_memory_db(args):
    db_memory = []
    path_to_db_data = args.path_to_data + '/' + args.valid_db_path

    with open(path_to_db_data, "r", encoding="utf-8") as file:
        lines = csv.reader(file, delimiter="\t", quotechar='"')

        for line in lines:
            if args.retrieval_mode == 'base':
                pair = line[:args.top_k]
            elif args.retrieval_mode == 'r_prime_sim':
                qr_pair = line[:args.top_k]
                pair = [value.split('[SEP]')[1].strip() for value in qr_pair]
            elif args.retrieval_mode == 'q_prime_sim':
                qr_pair = line[:args.top_k]
                pair = [value.split('[SEP]')[0].strip() for value in qr_pair]
            else:
                print("===No Valid Mode===\n===Please Check args.retrieval_mode===")
                exit()

            db_memory.append(pair)

    print(f"\nDB LENGTH (Training): {len(db_memory)}\n")

    return db_memory


def get_cur_db_data(args, tokenizer, search_idx, db_data, labels):
    cur_batch_db_data = []
    search_idx = search_idx.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()

    for step, idx in enumerate(search_idx):
        cur_db_data = db_data[idx[0]]

        cur_db_data = cur_db_data[labels[step][0]]
        cur_batch_db_data.append(cur_db_data)

    embeddings = tokenizer(cur_batch_db_data,
                           truncation=True,
                           return_tensors="pt",
                           max_length=args.max_len,
                           padding='max_length')

    return embeddings.to(args.device)


def get_top_db_data(args, tokenizer, db_data, idx):
    cur_idx_db_data = []

    for i in range(len(db_data)):
        cur_idx_db_data.append(db_data[i][idx])

    embeddings = tokenizer(cur_idx_db_data,
                           truncation=True,
                           return_tensors="pt",
                           max_length=args.max_len,
                           padding='max_length')

    return embeddings.to(args.device)


if __name__ == '__main__':
    get_loader('test')
