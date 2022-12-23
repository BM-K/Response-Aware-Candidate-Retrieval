import torch
from argparse import ArgumentParser

from transformers import (BartTokenizer,
                          BertTokenizer,
                          PreTrainedTokenizerFast,
                          BartForConditionalGeneration)

from models.retrieval import RetrievalModel
from models.generative import GenerativeModel

class Arguments():

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self):
        self.add_argument('--lang', type=str, default='ko')
        self.add_argument('--eval', type=str, default='True')
        self.add_argument('--end_command', type=str, default='-1')
        self.add_argument('--model', type=str, default='hybrid')
        self.add_argument('--use_kb', type=str, default='False')
        self.add_argument('--retrieval_mode', type=str, default='base')
        self.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def add_hyper_parameters(self):
        self.add_argument('--num_ks', type=int, default=3)
        self.add_argument('--ret_max_len', type=int, default=80)
        self.add_argument('--max_len', type=int, default=80)
        self.add_argument('--mips', type=str, default='False')
        self.add_argument('--num_centroids', type=int, default=64)
        self.add_argument('--db_embedding_bsz', type=int, default=512)
        self.add_argument('--n_beams', type=int, default=1)
        self.add_argument('--min_length', type=int, default=10)
        self.add_argument('--embedding_dim', type=int, default=768)

    def add_data_parameters(self):
        self.add_argument('--test_data', type=str, default='../REGE/data/base_labeled_test.tsv')
        self.add_argument('--corpus_train', type=str, default='../RACAR/data/labeled_train.tsv')
        self.add_argument('--corpus_valid', type=str, default='../RACAR/data/labeled_dev.tsv')
        self.add_argument('--retrieval_ckpt', type=str, default='../RACAR/output/retrieval_racar.pt')
        self.add_argument('--generative_ckpt', type=str, default='../REGE/output/generator_rege.pt')

        self.add_argument('--results_ref', type=str, default='inference_results_ref.tsv')
        self.add_argument('--results_hyp', type=str, default='inference_results_hyp.tsv')

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:print("\nargparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:print("\t", key, ":", value, "\n}")
            else:print("\t", key, ":", value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        self.print_args(args)

        return args

    def run(self):
        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_hyper_parameters()
        parser.add_data_parameters()

        return parser.parse()


class Setting():

    def __init__(self, args):
        self.args = args
        
        self.retrieval_model = RetrievalModel('klue/bert-base')
        self.retrieval_tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        
        self.generative_model = GenerativeModel("gogamza/kobart-base-v2", self.args)
        special_tokens = {'sep_token': "<sep>"}
        
        self.generative_tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
        self.generative_tokenizer.add_special_tokens(special_tokens)
        self.generative_model.model.resize_token_embeddings(len(self.generative_tokenizer.get_vocab()))
        
    def model(self,
        ):
        print("\n\t=====Model Setting=====")
        retrival_model, generative_model = self.load(self.retrieval_model,
                                                     self.generative_model)
        print("     ***Model Setting Complete***")
        return retrival_model, generative_model, self.retrieval_tokenizer, self.generative_tokenizer

    def load(self, 
             retrival_model=None,
             generative_model=None,
        ):
    
        retrival_model.to(self.args.device)
        generative_model.to(self.args.device)

        retrival_model.load_state_dict(torch.load(self.args.retrieval_ckpt))
        print("\t RETRIEVAL MODEL  [✔]")
        generative_model.load_state_dict(torch.load(self.args.generative_ckpt))
        print("\t GENERATIVE MODEL [✔]")

        return retrival_model.eval(), generative_model.eval()
