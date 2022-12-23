CUDA_VISIBLE_DEVICES=0 python end2end_inference.py --lang ko --model hybrid --num_ks 3 --max_len 80 --test_data ../REGE/data/base_labeled_test.tsv --corpus_train ../RACAR/data/labeled_train.tsv --corpus_valid ../RACAR/data/labeled_dev.tsv --retrieval_ckpt ../RACAR/output/retrieval_racar.pt --generative_ckpt ../REGE/output/generator_rege.pt --results_ref inference_results_ref.tsv --results_hyp inference_results_hyp.tsv --retrieval_mode base --num_centroids 64

python evaluation.py 
