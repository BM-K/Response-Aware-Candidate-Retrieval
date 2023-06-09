import csv
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, type_):
        self.args = args
        self.metric = metric
        self.type = type_

        self.tokenizer = tokenizer
        self.file_path = file_path

        self.input_ids = []
        self.attention_mask = []
        self.decoder_input_ids = []
        self.decoder_attention_mask = []
        self.labels = []
        self.bl_decoder_input_ids = []
        self.bl_decoder_attention_mask = []
        self.bl_labels = []

        special_tokens = {'sep_token': "<sep>"}
        self.tokenizer.add_special_tokens(special_tokens)

        self.init_token = self.tokenizer.bos_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.eos_token = self.tokenizer.eos_token
        self.sep_token = self.tokenizer.sep_token

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.sep_token_idx = self.tokenizer.convert_tokens_to_ids(self.sep_token)

        print(self.init_token, self.init_token_idx)
        print(self.pad_token, self.pad_token_idx)
        print(self.eos_token, self.eos_token_idx)
        print(self.sep_token, self.sep_token_idx)
        
        self.ignore_index = -100

    def load_data(self, type):
        st = 0
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')

            for _, line in tqdm(enumerate(lines)):
                check = self.data2tensor(line, type)
                st+=1

    def data2tensor(self, line, type):
        try:
            source, target, db_response = line[0].strip(), line[1].strip(), line[2]
            
            db_response = db_response.split('|')
            adam_knowledge = db_response[-1].strip()
            db_response = ' | '.join(db_response[:-1])
            
        except IndexError:
            logger.info("Index Error")
            exit()
            return False

        if self.args.model == 'BART':
            if self.args.use_db_response == 'True':

                """
                *** Fusion in Decoder Strategy ***
                """
                if self.args.n_fid > 1:
                    split_db_response = db_response.split('|')
                    split_db_response = split_db_response[:self.args.n_fid]
                    
                    if self.args.use_kb == 'True':
                        if adam_knowledge == 'None':
                            knowledge = ' '
                        else:
                            knowledge = f" {self.sep_token} {adam_knowledge}"

                    input_list = []
                    attention_mask_list = []
                    inputs = self.tokenizer.encode(source) + [self.sep_token_idx]
                                          
                    for step in range(len(split_db_response)):
                        
                        if self.args.use_kb == 'True':
                            if knowledge == ' ':
                                db_res = self.tokenizer.encode(split_db_response[step].strip())
                            else:
                                db_res = self.tokenizer.encode(split_db_response[step].strip()+knowledge)
                        else:
                            db_res = self.tokenizer.encode(split_db_response[step].strip())
                        
                        #db_res = self.tokenizer.encode(split_db_response[step].strip())

                        input = self.add_padding_data(inputs + db_res)

                        input_list.append(input)

                        attention_mask = torch.LongTensor(input).ne(self.pad_token_idx).float()
                        attention_mask_list.append(attention_mask.numpy())
                    
                    if len(split_db_response) < self.args.n_fid:
                        for i in range(self.args.n_fid-len(split_db_response)):
                            input_list.append(input)
                            attention_mask_list.append(attention_mask.numpy())
                        print("Less FiD")
                
                    input_list = torch.LongTensor(input_list)
                    attention_mask_list = torch.tensor(attention_mask_list)

                    self.input_ids.append(input_list)
                    self.attention_mask.append(attention_mask_list)
                else:
                    db_response_ = db_response.split('|')[0]

                    inputs = self.tokenizer.encode(source) + [self.sep_token_idx]
                    db_res = self.tokenizer.encode(db_response_)

                    inputs = inputs + db_res
                    inputs = torch.LongTensor(self.add_padding_data(inputs))
                    attention_mask = torch.tensor(inputs.ne(self.pad_token_idx).float())

                    self.input_ids.append(inputs)
                    self.attention_mask.append(attention_mask)

                # Alpha blending

                db_response = db_response.split('|')[0]

                blending_label_ids = self.add_ignored_data(self.tokenizer.encode(db_response) + [self.eos_token_idx])
                blending_label_ids = torch.LongTensor(blending_label_ids)

                blending_dec_input_ids = self.add_padding_data([self.init_token_idx]+ self.tokenizer.encode(db_response) + [self.eos_token_idx])
                blending_dec_input_ids = torch.LongTensor(blending_dec_input_ids)

                blending_decoder_attention_mask = torch.tensor(blending_dec_input_ids.ne(self.pad_token_idx).float())

                self.bl_labels.append(blending_label_ids)
                self.bl_decoder_input_ids.append(blending_dec_input_ids)
                self.bl_decoder_attention_mask.append(blending_decoder_attention_mask)

                label_ids = self.add_ignored_data(self.tokenizer.encode(target) + [self.eos_token_idx])
                label_ids = torch.LongTensor(label_ids)

                dec_input_ids = self.add_padding_data([self.init_token_idx] + self.tokenizer.encode(target) + [self.eos_token_idx])
                dec_input_ids = torch.LongTensor(dec_input_ids)

                decoder_attention_mask = torch.tensor(dec_input_ids.ne(self.pad_token_idx).float())

            else:
                #if self.args.use_kb == 'True':
                #    if adam_knowledge != 'None':
                #        knowledge = f" {self.sep_token} {adam_knowledge.strip().replace(' ', '')}"
                #        source = source + knowledge
                        
                inputs = self.tokenizer.encode(source)  # [:-1]

                inputs = torch.LongTensor(self.add_padding_data(inputs))
                attention_mask = torch.tensor(inputs.ne(self.pad_token_idx).float())

                label_ids = self.add_ignored_data(self.tokenizer.encode(target) + [self.eos_token_idx])
                label_ids = torch.LongTensor(label_ids)

                dec_input_ids = self.add_padding_data([self.init_token_idx] + self.tokenizer.encode(target) + [self.eos_token_idx])
                dec_input_ids = torch.LongTensor(dec_input_ids)

                decoder_attention_mask = torch.tensor(dec_input_ids.ne(self.pad_token_idx).float())

                self.input_ids.append(inputs)
                self.attention_mask.append(attention_mask)

            self.decoder_input_ids.append(dec_input_ids)
            self.decoder_attention_mask.append(decoder_attention_mask)
            self.labels.append(label_ids)

    def add_padding_data(self, inputs):
        if len(inputs) <= self.args.max_len:
            pad = np.array([self.pad_token_idx] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            if self.args.model == 'GPT':
                inputs = inputs[:self.args.max_len]
            else:
                inputs = inputs[:self.args.max_len-1] + [self.eos_token_idx]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) <= self.args.max_len:
            pad = np.array([self.ignore_index] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            if self.args.model == 'GPT':
                inputs = inputs[:self.args.max_len]
            else:
                inputs = inputs[:self.args.max_len-1] + [self.eos_token_idx]

        return inputs

    def __getitem__(self, index):
        if self.args.use_db_response == 'True':
            input_data = {'input_ids': self.input_ids[index].to(self.args.device),
                          'attention_mask': self.attention_mask[index].to(self.args.device),
                          'decoder_input_ids': self.decoder_input_ids[index].to(self.args.device),
                          'decoder_attention_mask': self.decoder_attention_mask[index].to(self.args.device),
                          'labels': self.labels[index].to(self.args.device),
                          'bl_decoder_input_ids': self.bl_decoder_input_ids[index].to(self.args.device),
                          'bl_decoder_attention_mask': self.bl_decoder_attention_mask[index].to(self.args.device),
                          'bl_labels': self.bl_labels[index].to(self.args.device)}
        else:
            input_data = {'input_ids': self.input_ids[index].to(self.args.device),
                          'attention_mask': self.attention_mask[index].to(self.args.device),
                          'decoder_input_ids': self.decoder_input_ids[index].to(self.args.device),
                          'decoder_attention_mask': self.decoder_attention_mask[index].to(self.args.device),
                          'labels': self.labels[index].to(self.args.device)}

        return input_data

    def __len__(self):
        return len(self.input_ids)


def get_loader(args, metric):
    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.test_data

    if args.model == 'BART':
        tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer, 'train')
        valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer, 'valid')
        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer, 'test')
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=False)}

    else:
        logger.info("Error: None type loader")
        exit()

    return loader, tokenizer

if __name__ == '__main__':
    get_loader('test')
