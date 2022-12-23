import logging
import torch.cuda.amp as amp
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
from trainer.loss import Loss
from trainer.utils import Metric
from transformers import get_linear_schedule_with_warmup
from trainer.semantic_similarity_model.models import ScoreModel
from data.dataloader import get_loader, get_cur_db_data, post_memory_db, test_labels

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.hypo = []

        self.args = args
        self.config = None

        self.loss_fn = Loss(args)
        self.metric = Metric(args)

        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': 0, 'iter': 0, 'acc': 0, 'f1': 0}

        self.sorted_path = args.path_to_save + args.ckpt
        self.post_memory_db = post_memory_db(self.args)

        self.test_labels = test_labels(self.args)

    def run(self, inputs, epoch=-1, mode='Normal'):

        post_logits, prior_logits, labels = self.config['model'](inputs,
                                                                self.post_memory_db,
                                                                mode)

        if mode == 'train' or mode == 'valid':
            if self.args.retrieval_mode == 'base':
                post_loss = self.loss_fn.base(self.config, post_logits, labels)
                prior_loss = self.loss_fn.base(self.config, prior_logits, labels)
                kd_loss = self.loss_fn.kd_loss(self.config, post_logits, prior_logits)
                loss = post_loss + prior_loss + kd_loss
            else:
                prior_loss = self.loss_fn.base(self.config, prior_logits, labels)
                loss = prior_loss

            acc, f1 = self.metric.cal_performance(prior_logits,
                                                  labels,
                                                  self.hypo,
                                                  mode=mode)

            return loss, acc, f1

        else:
            hit_10, hit_k, mrr_k = self.metric.cal_performance(prior_logits,
                                                               labels,
                                                               self.hypo,
                                                               mode=mode)

            return hit_10, hit_k, mrr_k

    def progress(self, loss, acc, f1):
        self.model_progress['iter'] += 1
        self.model_progress['loss'] += loss
        self.model_progress['acc'] += acc
        self.model_progress['f1'] += f1

    def return_value(self, mode='Normal'):
        if mode == 'Normal':
            loss = self.model_progress['loss'].cpu().numpy() / self.model_progress['iter']
            acc = self.model_progress['acc'] / self.model_progress['iter']
            f1 = self.model_progress['f1'] / self.model_progress['iter']
            return loss, acc, f1, 0
        else:
            loss = self.model_progress['loss'] / self.model_progress['iter']
            acc = self.model_progress['acc'] / self.model_progress['iter']
            f1 = self.model_progress['f1'] / self.model_progress['iter']
            return loss, acc, f1, 0

    def get_object(self, tokenizer, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=self.args.warmup_ratio * train_total,
            num_training_steps=train_total)

        return scheduler

    def model_setting(self):
        loader, tokenizer = get_loader(self.args, self.metric)
        scaler = amp.GradScaler()

        model = ScoreModel(self.args, tokenizer)
        model.to(self.args.device)

        criterion, optimizer = self.get_object(tokenizer, model)

        if self.args.test == 'False':
            scheduler = self.get_scheduler(optimizer, loader['train'])
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'scaler': scaler,
                  'args': self.args,
                  'model': model}

        self.config = config

        return self.config

    def train(self, epoch):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):

            inputs = batch
            self.config['optimizer'].zero_grad()

            if self.args.fp16 == 'True':
                with amp.autocast():
                    loss, acc, f1 = self.run(inputs, epoch=epoch, mode='train')
                    
                self.config['scaler'].scale(loss).backward()
                self.config['scaler'].step(self.config['optimizer'])
                self.config['scaler'].update()
                self.config['scheduler'].step()

            else:
                loss, acc, f1 = self.run(inputs, epoch=epoch, mode='train')
                loss.backward()
                self.config['optimizer'].step()
                self.config['scheduler'].step()
    
            self.progress(loss.data, acc.data, f1)

        return self.return_value()

    def valid(self, epoch):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):
                inputs = batch
                loss, acc, f1 = self.run(inputs, epoch=epoch, mode='valid')

                self.progress(loss.data, acc.data, f1)

        return self.return_value()

    def test(self):
        self.config['model'].load_state_dict(torch.load(self.sorted_path))
        self.config['model'].eval()
        self.config['model'].get_db_data.embedding_db_data()

        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.config['loader']['test'])):
                inputs = batch

                if self.args.test_data.find('test') != -1:
                    inputs['test_labels'] = torch.tensor(self.test_labels)

                hit_10, hit_args, mrr_k = self.run(inputs, mode='test')

                self.progress(hit_10, hit_args, mrr_k)

        self.metric.db_Rprime_generation(self.hypo)

        return self.return_value(mode='test')
