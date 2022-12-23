import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT2LMHead(nn.Module):
    def __init__(self, args, tokenizer):
        super(GPT2LMHead, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.vocab_size = self.gpt2.config.vocab_size
        self.args = args

    def forward(self, inputs, mode):
        if mode != 'test':
            outputs = self.gpt2(input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                labels=inputs['labels'])
            return outputs.loss

        else:
            outputs = self.gpt2.generate(inputs['input_ids'],
                                         max_length=self.args.max_len+1,
                                         num_beams=5)
            return outputs