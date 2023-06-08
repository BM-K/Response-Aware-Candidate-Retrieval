import os
import csv
import torch
import logging
import itertools

from rouge import Rouge
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, recall_score, precision_score


logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.step = 0
        self.args = args
        self.rouge = Rouge()
        self.rouge_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
        self.hits = 0

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def remove_duple(self, rank):
        previous_val = 0
        removed_rank = torch.randn(1, 2)
        for step, idx in enumerate(range(len(rank))):

            if step == 0:
                previous_val = rank[step][0]
                removed_rank = torch.cat([removed_rank, rank[step].view(1, -1)], dim=0)
            else:
                if previous_val == rank[step][0]: continue
                else:
                    previous_val = rank[step][0]
                    removed_rank = torch.cat([removed_rank, rank[step].view(1, -1)], dim=0)

        removed_rank = removed_rank[1:, -1] + 1

        return removed_rank.float()

    def cal_mrr(self, y, yhat, k=10):
        yhat = yhat.topk(k=k, dim=-1)[1].cpu()
        targets = y.unsqueeze(1)
        hits = (targets == yhat).nonzero()[:, -1] + 1
        ranks = torch.reciprocal(hits.float())
        mrr_score = torch.sum(ranks).data / y.size(0)

        return mrr_score

    def cal_hit(self, y, yhat, k=10):
        yhat = yhat.topk(k=k, dim=-1)[1].cpu()

        targets = y
        hits = (targets == yhat).nonzero()

        if len(hits) == 0:
            return 0

        n_hits = (targets == yhat).nonzero()[:, :-1].size(0)
        hit_score = float(n_hits) / y.size(0)

        return hit_score

    def cal_rec(self, preds, target):
        score = 0
        k = self.args.recall_k
        for i in range(len(target)):
            cur_t = target[i:i+1, :].squeeze(0)
            cur_p = preds[i:i+1, :].squeeze(0)

            relevant = cur_t[torch.argsort(cur_p, dim=-1, descending=True)][:k].sum().float()
            score += relevant / cur_t.sum()

        score /= len(target)

        return score

    def cal_performance(self, yhat, y, hypo, mode='Normal'):
        with torch.no_grad():
            if mode == 'train' or mode == 'valid':
                y = y.cpu()
                yhat = yhat.max(dim=-1)[1].cpu()

                acc = (yhat == y).float().mean()
                f1 = f1_score(y, yhat, average='macro')

                return acc, f1

            else:
                yhat, y, preds, target = y  # test time y = (yhat, y, preds, target for recall)

                if y == None:
                    hit_10 = hit_args = mrr_k = 0
                else:
                    hit_args = self.cal_hit(y.cpu(), yhat.cpu(), k=self.args.hit_k)
                    mrr_k = self.cal_mrr(y.cpu(), yhat.cpu(), k=self.args.hit_k)
                    hit_10 = self.cal_hit(y.cpu(), yhat.cpu(), k=10)

                yhat = yhat.topk(k=self.args.retrieve_n_response, dim=-1)[1].cpu()
                hypo.append(yhat[:, :self.args.retrieve_n_response].numpy().tolist())  # edit for FiD
            
                return hit_10, hit_args, mrr_k

    def db_Rprime_generation(self, checker):
        query_set = []
        response_set = []
        db_response_set = []
        path_to_db_data_train = self.args.path_to_data + '/' + self.args.train_data
        path_to_db_data_valid = self.args.path_to_data + '/' + self.args.valid_data
        path_to_db_data = {'train': path_to_db_data_train, 'valid': path_to_db_data_valid}
        path_to_test_data = self.args.path_to_data + '/' + self.args.test_data

        for key, val in path_to_db_data.items():
            with open(path_to_db_data[key], "r", encoding="utf-8") as file:
                lines = csv.reader(file, delimiter="\t", quotechar='"')

                for line in lines:
                    _, response = line[0].strip(), line[1].strip()
                    db_response_set.append(response)

        with open(path_to_test_data, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')

            for line in lines:
                query, response = line[0].strip(), line[1].strip()
                query_set.append(query)
                response_set.append(response)

        checker = list(itertools.chain.from_iterable(checker))
        query_response_pair_for_dialogue_generation = []
        for idx in range(len(checker)):

            gold_response = response_set[idx]
            # hypo = db_response_set[checker[idx]]  # memory_db[search_idx][yhat].split('[SEP]')[1].strip()
            hypo = []
            for i in range(len(checker[idx])):
                hypo.append(db_response_set[checker[idx][i]])

            query_response_pair_for_dialogue_generation.append((query_set[idx], gold_response, hypo))

            self.step += 1

        dialogue_generation_data_path = self.args.path_to_data + '/' + self.args.retrieval_mode + '_' + self.args.test_data
        with open(dialogue_generation_data_path, "w", encoding="utf-8") as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            for line in query_response_pair_for_dialogue_generation:
                query, gold_response, response = line
                response = ' | '.join(response)
                tsv_writer.writerow([query.strip(), gold_response.strip(), response])

    def avg_rouge(self):
        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] /= self.step

        return self.rouge_scores

    def performance_check(self, cp):
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Valid Loss: {cp["vl"]:.4f}==')
        print(f'\t==Train Acc: {cp["ta"]:.4f} | Valid Acc: {cp["va"]:.4f}==')
        print(f'\t==Train F1: {cp["tf"]:.4f} | Valid F1: {cp["vf"]:.4f}==\n')

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def save_config(self, cp):
        config = "Config>>\n"
        for idx, (key, value) in enumerate(self.args.__dict__.items()):
            cur_kv = str(key) + ': ' + str(value) + '\n'
            config += cur_kv
        config += 'Epoch: ' + str(cp["ep"]) + '\t' + 'Valid loss: ' + str(cp['vl']) + '\n'

        with open(self.args.path_to_save+self.args.ckpt.split('.')[0]+'_config.txt', "w") as f:
            f.write(config)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + config['args'].ckpt

        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']
            torch.save(config['model'].state_dict(), sorted_path)
            self.save_config(cp)
            print(f'\n\t## SAVE Valid Loss: {cp["vl"]:.4f} ##')

        else:
            pco['early_stop_patient'] += 1
            if pco['early_stop_patient'] == config['args'].patient:
                pco['early_stop'] = True

        self.performance_check(cp)
