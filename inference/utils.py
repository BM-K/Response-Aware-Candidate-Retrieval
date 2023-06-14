import h5py
import torch
import jieba
import numpy as np

def cal_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def add_padding_data(inputs, max_len, pad_token_idx, eos_token_idx):
    if len(inputs) <= max_len:
        pad = np.array([pad_token_idx] * (max_len - len(inputs)))
        inputs = np.concatenate([inputs, pad])
    else:
        inputs = inputs[:max_len-1] + [eos_token_idx]

    return inputs

def h5py_io(file_name, embedded_db, mode):
    if mode == 'w':
        file_ = h5py.File(file_name, mode)
        group = file_.create_group('offline_embedded')
        group.create_dataset('embedded_db', data=embedded_db.cpu().detach().numpy())
        file_.close()
    else:
        hf = h5py.File(file_name, mode)
        embedded_db = torch.tensor(np.array(hf['offline_embedded'].get('embedded_db')))
        hf.close()

        return embedded_db

def output_results(args, test_ref_set, hyp_set, generative_tokenizer):
    with open(args.results_ref, 'w', encoding='utf-8') as f:
        for idx, ref in enumerate(test_ref_set):
            if args.lang == 'ko' or args.lang == 'en':
                ref = ' '.join(generative_tokenizer.tokenize(ref))
                if args.lang == 'ko':
                    ref = ref.replace('▁', '') + "\n"
                elif args.lang == 'en':
                    ref = ref.replace('Ġ', '') + "\n"
            else:
                ref = ' '.join(jieba.lcut(ref.replace(' ', ''))).strip() + '\n'
                
            f.write(ref)
            
    with open(args.results_hyp, 'w', encoding='utf-8') as f:
        for hyp in hyp_set:
            if args.lang == 'ko' or args.lang == 'en':
                hyp= ' '.join(generative_tokenizer.tokenize(hyp))
            
                if args.lang == 'ko':
                    hyp = hyp.replace('▁', '') + "\n"
                elif args.lang == 'en':
                    hyp = hyp.replace('Ġ', '') + "\n"
            else:
                hyp = ' '.join(jieba.lcut(hyp.replace(' ', ''))).strip() + '\n'

            f.write(hyp)

