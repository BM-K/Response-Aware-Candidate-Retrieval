import re
import os
import csv
import torch
import logging
from rouge import Rouge
from collections import Counter
from lib import evaluation_utils
from multiprocessing import Pool
from tensorboardX import SummaryWriter
import nltk.translate.bleu_score as bleu
from distinct_n import distinct_n_sentence_level
from nltk.translate.bleu_score import SmoothingFunction

logger = logging.getLogger(__name__)
writer = SummaryWriter()
chencherry = SmoothingFunction()


class Metric():

    def __init__(self, args):
        self.args = args
        self.step = 0
        self.dist = {'score1': 0, 'score2': 0}
        self.bleu_score = {'score1': 0,
                           'score2': 0,
                           'score3': 0,
                           'score4': 0}

        self.response = []
        self.get_reference()

        self.rouge = Rouge()
        self.rouge_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-l': {'r': 0, 'p': 0, 'f': 0}}

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def draw_graph(self, cp):
        writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])

    def performance_check(self, cp):
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Valid Loss: {cp["vl"]:.4f}==')

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

        with open(self.args.path_to_save + 'config.txt', "w") as f:
            f.write(config)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + config['args'].ckpt

        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']

            unwrapped_model = config['accelerator'].unwrap_model(config['model'])
            config['accelerator'].save(unwrapped_model.state_dict(), sorted_path)
             
            self.save_config(cp)
            print(f'\n\t## SAVE valid_loss: {cp["vl"]:.4f} ##')
        else:
            pco['early_stop_patient'] += 1
            if pco['early_stop_patient'] == config['args'].patient:
                pco['early_stop'] = True
                writer.close()

        # self.draw_graph(cp)
        self.performance_check(cp)

    def get_reference(self, ):
        path_to_test_data = self.args.path_to_data + self.args.test_data
        with open(path_to_test_data, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')

            for line in lines:
                query, response = line[0].strip(), line[1].strip()
                self.response.append(response)

    def result_file(self, config, hyp):
        sorted_path = config['args'].path_to_save + 'result.csv'
        with open(sorted_path, 'a', encoding='utf-8') as f:
            tw = csv.writer(f)
            if self.step == 0:
                tw.writerow(['id', 'summary'])
            tw.writerow([str(self.step + 1), hyp])

    def rouge_score(self, config, hyp, ref):
        try:
            score = self.rouge.get_scores(hyp, ref)[0]
        except ValueError:
            score = self.rouge.get_scores('a', 'a')[0]

        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] += score[metric][key]

        self.step += 1

    def avg_score(self):

        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] /= self.step

        for metric, scores in self.bleu_score.items():
            self.bleu_score[metric] /= self.step

        for metric, scores in self.dist.items():
            self.dist[metric] /= self.step

        return self.rouge_scores, self.bleu_score, self.dist

    def dialogue_f1(self, ref_hyp_dict):
        gold_items = ref_hyp_dict['ref']
        pred_items = ref_hyp_dict['hyp']

        assert len(gold_items) == len(pred_items)

        f1_score = 0.0
        iter_ = 0
        for gold_item, pred_item in zip(gold_items, pred_items):
            iter_ += 1
            gold_item = gold_item.strip().split(' ')
            pred_item = pred_item.strip().split(' ')

            common = Counter(gold_item) & Counter(pred_item)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(pred_item)
            recall = 1.0 * num_same / len(gold_item)
            f1_score += (2 * precision * recall) / (precision + recall)

        return (f1_score / iter_) * 100

    def print_bleu_score(self, save_ref_hyp):
        path_to_save_test_hyp = self.args.path_to_save + self.args.ckpt.split('.')[0] + '_test_hyp.txt'
        path_to_save_test_ref = self.args.path_to_save + self.args.ckpt.split('.')[0] + '_test_ref.txt'
        path_to_ref_src_file = self.args.path_to_data + self.args.train_data
        save_path = {'ref': path_to_save_test_ref, 'hyp': path_to_save_test_hyp, 'ref_src_file': path_to_ref_src_file}

        for key, value in save_ref_hyp.items():
            with open(save_path[key], "w", encoding="utf-8") as file:
                for src in value: file.write(src)

        f1_score = self.dialogue_f1(save_ref_hyp)

        hparams = {'test_tgt_file': save_path['ref'], 'ref_src_file': save_path['ref_src_file']}

        top1_out_file_path = save_path['hyp']
        metrics = 'bleu-1,bleu-2,bleu-3,bleu-4,distinct-1,distinct-2,entropy'.split(',')

        thread = 16
        thread_pool = Pool(thread)

        jobs = []
        scores = []
        for metric in metrics:
            if metric == 'entropy':
                job = thread_pool.apply_async(evaluation_utils.evaluate, (
                    hparams['test_tgt_file'],
                    top1_out_file_path,
                    hparams['ref_src_file'],
                    metric))
            else:
                job = thread_pool.apply_async(evaluation_utils.evaluate, (
                    hparams['test_tgt_file'],
                    top1_out_file_path,
                    None,
                    metric))
            jobs.append(job)

        print('\n=== Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py ===')
        for job, metric in zip(jobs, metrics):
            complex_score = job.get()
            score = complex_score[0:len(complex_score) // 2]
            if len(score) == 1:
                score = score[0]

            print(('\n\t%s -> %s') % (metric, score))

            if type(score) is list or type(score) is tuple:
                for x in score:
                    scores.append(str(x))
            else:
                scores.append(str(score))
        print(('\n\t%s -> %s') % ('F1 Score', f1_score))
        print('\n===================================================================================')

    def cal_distinct(self, config, hyp):
        hypothesis = [hyp.split(' ')]

        dist_1 = [distinct_n_sentence_level(s, 1) for s in hypothesis]
        dist_2 = [distinct_n_sentence_level(s, 2) for s in hypothesis]

        self.dist['score1'] += dist_1[0]
        self.dist['score2'] += dist_2[0]

    def cal_bleu_score(self, config, hyp, ref):
        hyp = hyp.split(' ')
        ref = ref.split(' ')

        self.bleu_score['score1'] += bleu.sentence_bleu([ref], hyp,
                                                        weights=(1., 0, 0, 0)) * 100
        self.bleu_score['score2'] += bleu.sentence_bleu([ref], hyp,
                                                        weights=(1. / 2, 1. / 2, 0, 0)) * 100
        self.bleu_score['score3'] += bleu.sentence_bleu([ref], hyp,
                                                        weights=(1. / 3, 1. / 3, 1. / 3, 0)) * 100
        self.bleu_score['score4'] += bleu.sentence_bleu([ref], hyp,
                                                        weights=(1. / 4, 1. / 4, 1. / 4, 1. / 4)) * 100
        #print(bleu.sentence_bleu([ref], hyp,
        #    weights=(1. , 0, 0, 0)) * 100)
        #print("=========================")
    def generation(self, config, save_ref_hyp, inputs):
        outputs = config['model'](inputs, mode='test')

        for step, beam in enumerate(outputs):
            ref = self.response[self.step]
            hyp = config['tokenizer'].decode(beam, skip_special_tokens=True)

            ref = ' '.join(config['tokenizer'].tokenize(ref))
            hyp = ' '.join(config['tokenizer'].tokenize(hyp))
            
            ref = ref.replace('▁', '').replace('o', '')
            hyp = hyp.replace('▁', '').replace('o', '')
            #print(ref)
            #print(hyp)
            save_ref_hyp['ref'].append(ref.strip() + "\n")
            save_ref_hyp['hyp'].append(hyp.strip() + "\n")

            self.rouge_score(config, hyp.strip(), ref.strip())
            self.cal_bleu_score(config, hyp.strip(), ref.strip())
            self.cal_distinct(config, hyp.strip())
        #exit()

