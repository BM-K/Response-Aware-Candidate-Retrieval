#!/bin/bash

#CUDA_VISIBLE_DEVICES=1 python train.py --fp16 False --use_db_response False --train True --test False --ckpt no_use_db_food.pt --max_len 80 --multi_gpu False
#CUDA_VISIBLE_DEVICES=1 python train.py --fp16 True --use_db_response False --train False --test True --ckpt no_use_db_food.pt --multi_gpu False  --max_len 80

#CUDA_VISIBLE_DEVICES=0 python train.py --fp16 False --multi_gpu False --use_db_response True --use_kb True --train True --test False --ckpt base_kb_fid3.pt --train_data base_kb_new_train.tsv --test_data base_kb_new_test.tsv --valid_data base_kb_new_valid.tsv --n_fid 3 --alpha_blending 0.0 --max_len 100
#CUDA_VISIBLE_DEVICES=0 python train.py --fp16 True --batch_size 128 --use_db_response True --use_kb True --train False --test True --ckpt base_kb_fid3.pt --train_data base_kb_new_train.tsv --test_data base_kb_new_test.tsv --valid_data base_kb_new_valid.tsv --n_fid 3 --alpha_blending 0.0 --max_len 100 --num_beams 1

#CUDA_VISIBLE_DEVICES=0 python train.py --fp16 False --multi_gpu False --use_db_response True --train True --test False --ckpt q_food_f7.pt --train_data q_prime_sim_food_train.tsv --test_data q_prime_sim_food_test.tsv --valid_data q_prime_sim_food_valid.tsv --n_fid 7 --alpha_blending 0.0 --max_len 80
#CUDA_VISIBLE_DEVICES=0 python train.py --fp16 True --use_db_response True --train False --test True --ckpt q_food_f5.pt --batch_size 128 --train_data q_prime_sim_food_train.tsv --test_data q_prime_sim_food_test.tsv --valid_data q_prime_sim_food_valid.tsv  --n_fid 5 --alpha_blending 0.0 --max_len 80 --num_beams 1

#CUDA_VISIBLE_DEVICES=0 python train.py --fp16 False --multi_gpu False --use_db_response True --train True --test False --ckpt r_food_f7.pt --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv --n_fid 7 --alpha_blending 0.0 --max_len 80
#CUDA_VISIBLE_DEVICES=1 python train.py --fp16 True --use_db_response True --train False --test True --ckpt r_food_f5.pt --batch_size 128 --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv  --n_fid 5 --alpha_blending 0.0 --max_len 80 --num_beams 1

#CUDA_VISIBLE_DEVICES=1 python train.py --multi_gpu False --fp16 False --use_db_response True --train True --test False --ckpt r_food_f1.pt --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv --n_fid 1 --alpha_blending 0.0 --max_len 64
#CUDA_VISIBLE_DEVICES=1 python train.py --fp16 True --use_db_response True --train False --test True --ckpt r_food_f1.pt --batch_size 128 --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv  --n_fid 1 --alpha_blending 0.0 --max_len 64

#CUDA_VISIBLE_DEVICES=1 python train.py --multi_gpu False --fp16 False --use_db_response True --train True --test False --ckpt r_food_f3.pt --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv --n_fid 3 --alpha_blending 0.0 --max_len 64
#CUDA_VISIBLE_DEVICES=1 python train.py --fp16 True --use_db_response True --train False --test True --ckpt r_food_f3.pt --batch_size 128 --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv  --n_fid 3 --alpha_blending 0.0 --max_len 64

CUDA_VISIBLE_DEVICES=1 python train.py --batch_size 32 --fp16 False --multi_gpu False --use_db_response True --train True --test False --ckpt generator_rege.pt --train_data base_labeled_train.tsv --test_data base_labeled_test.tsv --valid_data base_labeled_dev.tsv --n_fid 3 --alpha_blending 0.0 --max_len 100
CUDA_VISIBLE_DEVICES=1 python train.py --fp16 True --batch_size 128 --use_db_response True --train False --test True --ckpt generator_rege.pt --train_data base_labeled_train.tsv --test_data base_labeled_test.tsv --valid_data base_labeled_dev.tsv --n_fid 3 --alpha_blending 0.0 --max_len 100

#CUDA_VISIBLE_DEVICES=1 python train.py --fp16 False --multi_gpu False --use_db_response True --train True --test False --ckpt q_food_f5.pt --train_data q_prime_sim_food_train.tsv --test_data q_prime_sim_food_test.tsv --valid_data q_prime_sim_food_valid.tsv --n_fid 5 --alpha_blending 0.0 --max_len 100
#CUDA_VISIBLE_DEVICES=1 python train.py --fp16 True --use_db_response True --train False --test True --ckpt q_food_f5.pt --batch_size 128 --train_data q_prime_sim_food_train.tsv --test_data q_prime_sim_food_test.tsv --valid_data q_prime_sim_food_valid.tsv  --n_fid 5 --alpha_blending 0.0 --max_len 100

#CUDA_VISIBLE_DEVICES=1 python train.py --multi_gpu False --use_db_response True --train True --test False --ckpt r_food_f3.pt --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv --n_fid 3 --alpha_blending 0.0 --lr 0.00002 --max_len 100
#CUDA_VISIBLE_DEVICES=1 python train.py --use_db_response True --train False --test True --ckpt r_food_f3.pt --batch_size 128 --train_data r_prime_sim_food_train.tsv --test_data r_prime_sim_food_test.tsv --valid_data r_prime_sim_food_valid.tsv  --n_fid 3 --alpha_blending 0.0 --max_len 100
