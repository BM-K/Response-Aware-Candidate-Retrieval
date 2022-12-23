import csv
import time
import torch
import faiss
from utils import *
from random import *
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances

class DialogueGeneration():

    def __init__(self, 
                 args=None,
                 retrieval_model=None,
                 generative_model=None,
                 retrieval_tokenizer=None,
                 generative_tokenizer=None,
                 embedded_database=None,
                 corpus=None,
                 faiss_index=None,
        ):

        self.args = args
        
        self.retrieval_model = retrieval_model
        self.generative_model = generative_model

        self.retrieval_tokenizer = retrieval_tokenizer
        self.generative_tokenizer = generative_tokenizer

        self.embedded_database = embedded_database
        self.corpus = corpus

        self.faiss_index = faiss_index
        
        self.pad_idx = generative_tokenizer.convert_tokens_to_ids(generative_tokenizer.pad_token)
        self.eos_idx = generative_tokenizer.convert_tokens_to_ids(generative_tokenizer.eos_token)
        self.sep_idx = generative_tokenizer.convert_tokens_to_ids(generative_tokenizer.sep_token)
    
    def user_inference(self,
        ):
        print("\nLet's start!")
        print(f"If you want to quit the conversation, please type \"{self.args.end_command}\".\n")
        
        retrieved_responses = []
        generative_model_inputs = []

        with torch.no_grad():
            while True:
                utter = input("You: ")
                if utter == self.args.end_command:
                    print("Bot: Good bye.")
                    break

                tokenized_input = self.retrieval_tokenizer(utter,
                                                           truncation=True,
                                                           return_tensors="pt",
                                                           max_length=self.args.ret_max_len,
                                                           padding='max_length')
                tokenized_input.to(self.args.device)
            
                embedded_input, _ = self.retrieval_model(tokenized_input, response=True)
                embedded_input = embedded_input[:, :1]

                if self.args.mips == 'False':
                    _, I = self.faiss_index.search(embedded_input.squeeze(0).cpu().detach().numpy(), self.args.num_ks)
                    preds = torch.tensor(I).squeeze(0)
                else:
                    logits = torch.matmul(embedded_input.squeeze(0),
                                          self.embedded_database.to(self.args.device).transpose(-1, -2))
                    preds = logits.topk(k=self.args.num_ks, dim=-1)[-1].squeeze(0)
                
                inputs = self.generative_tokenizer.encode(utter) + [self.sep_idx]

                print("\n-----검색결과-----")
                for step, idx in enumerate(preds):
                    proactive_response = self.generative_tokenizer.encode(self.corpus[idx].split('[SEP]')[-1])
                    print(f"{step+1}.{self.corpus[idx].split('[SEP]')[-1]}")
                    input_ = add_padding_data(inputs + proactive_response,
                                              self.args.max_len,
                                              self.pad_idx,
                                              self.eos_idx)
                    generative_model_inputs.append(input_)
                print("------------------\n")
            
                input_list = torch.LongTensor(generative_model_inputs).to(self.args.device).unsqueeze(0)
                outputs = self.generative_model(input_list)

                hyp = self.generative_tokenizer.decode(outputs.squeeze(0), skip_special_tokens=True)
                print(f'Bot: {hyp}\n')

    def eval_inference(self,
        ):
        candidate_response = []
        
        test_set = []
        test_ref_set = []
        hyp_set = []
        
        
        with open(self.args.test_data, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')
            for line in lines:
                test_set.append(line[0])        
                test_ref_set.append(line[1])
        
        start_time = time.time()        
        for ii, data in enumerate(tqdm(test_set)):
        
            input_list = []
            
            if self.args.model == 'hybrid':
                tokenized_input = self.retrieval_tokenizer(data,
                                                           truncation=True,
                                                           return_tensors="pt",
                                                           max_length=self.args.ret_max_len,
                                                           padding='max_length')

                tokenized_input.to(self.args.device)

                embedded_input, _ = self.retrieval_model(tokenized_input, response=True)
                embedded_input = embedded_input[:, :1]
                embedded_input = embedded_input.squeeze(0).cpu().detach().numpy()

                if self.args.mips == 'False':
                    scores, I = self.faiss_index.search(embedded_input, self.args.num_ks)
                    preds = torch.tensor(I).squeeze(0)
                else:
                    logits = torch.matmul(embedded_input.squeeze(0),
                                          self.embedded_database.to(self.args.device).transpose(-1, -2))
                    preds = logits.topk(k=self.args.num_ks, dim=-1)[-1].squeeze(0)
                
                if self.args.lang == 'zh':
                    inputs = self.generative_tokenizer.encode(data)[:-1] + [self.sep_idx]
                else:
                    inputs = self.generative_tokenizer.encode(data) + [self.sep_idx]

                for step, idx in enumerate(preds):
                    if step == 0:
                        candidate_response.append(self.corpus[idx].split('[SEP]')[-1])
                    if self.args.lang == 'ko':
                        proactive_response = self.generative_tokenizer.encode(self.corpus[idx].split('[SEP]')[-1].strip())
                    else:
                        proactive_response = self.generative_tokenizer.encode(self.corpus[idx].split('[SEP]')[-1])[1:-1]                
                    
                    input_ = add_padding_data(inputs + proactive_response,
                                              self.args.max_len,
                                              self.pad_idx,
                                              self.eos_idx)
                    input_list.append(input_)

            else:
                inputs = self.generative_tokenizer.encode(data)
                input_list.append(add_padding_data(inputs,
                                                   self.args.max_len,
                                                   self.pad_idx,
                                                   self.eos_idx))

            input_list = torch.LongTensor(input_list).to(self.args.device).unsqueeze(0)
            outputs = self.generative_model(input_list)

            hyp = self.generative_tokenizer.decode(outputs.squeeze(0), skip_special_tokens=True)
            hyp_set.append(hyp)
            
        end_time = time.time()
        epoch_mins, epoch_secs = cal_time(start_time, end_time)
        
        fn = self.args.results_hyp + ".txt"
        with open(fn ,'w', encoding='utf-8') as f:
            for r in candidate_response:
                r = r.strip() + '\n'
                f.write(r)

        output_results(self.args, test_ref_set, hyp_set, self.generative_tokenizer)
        print(f"\nInference Time: {epoch_mins}m {epoch_secs}s")
    
