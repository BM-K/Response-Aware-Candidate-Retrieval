import logging
import torch.nn as nn
import torch.quantization
import torch.optim as optim

from apex import amp
from tqdm import tqdm
from trainer.utils import Metric
from accelerate import Accelerator
from data.dataloader import get_loader
from trainer.GenerationModel.GPT2 import GPT2LMHead
from transformers import get_linear_schedule_with_warmup
from trainer.GenerationModel.BART import BARTConditionalGeneration
from trainer.GenerationModel.Blenderbot import BlenderbotConditionalGeneration

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.save_ref_hyp = {'ref': [], 'hyp': []}
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': -1, 'iter': -1, 'acc': -1}
        self.sorted_path = args.path_to_save + args.ckpt

    def run(self, inputs, mode=None):
        loss = self.config['model'](inputs, mode)
        return loss

    def progress(self, loss):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']

        return loss

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
        accelerator = Accelerator(fp16=True)

        loader, tokenizer = get_loader(self.args, self.metric)

        if self.args.model == 'BART':
            model = BARTConditionalGeneration(self.args, tokenizer)

        if self.args.multi_gpu == 'True':
            model = nn.DataParallel(model, output_device=0)
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
                  'args': self.args,
                  'accelerator': accelerator,
                  'model': model}

        if config['args'].fp16 == 'True' and config['args'].test == 'True' and config['args'].multi_gpu == 'False':
            config['model'], config['optimizer'] = amp.initialize(
                config['model'], config['optimizer'], opt_level=config['args'].opt_level)
        else:
            config['model'], config['optimizer'] = accelerator.prepare(model, optimizer)

        self.config = config

        return self.config

    def train(self):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        train_loader = tqdm(self.config['accelerator'].prepare(self.config['loader']['train']),
                            desc='Loading train dataset')
        for step, batch in enumerate(train_loader):

            self.config['optimizer'].zero_grad()

            inputs = batch
            loss = self.run(inputs, mode='train')
            loss = torch.mean(loss)

            if self.args.fp16 == 'True' and self.args.multi_gpu == 'False':
                with amp.scale_loss(loss, self.config['optimizer']) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.config['accelerator'].backward(loss)

            self.config['optimizer'].step()
            self.config['scheduler'].step()
            self.progress(loss.data)

            train_loader.set_description("Loss %.04f | step %d" % (loss.data, step))

        return self.return_value()

    def valid(self):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        valid_loader = self.config['accelerator'].prepare(self.config['loader']['valid'])
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                inputs = batch
                loss = self.run(inputs, mode='valid')
                self.progress(torch.mean(loss).data)

        return self.return_value()

    def test(self):
        self.config['model'].load_state_dict(torch.load(self.sorted_path))
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.config['loader']['test'])):
                inputs = batch
                self.metric.generation(self.config, self.save_ref_hyp, inputs)

        self.metric.print_bleu_score(self.save_ref_hyp)

        return self.metric.avg_score()
