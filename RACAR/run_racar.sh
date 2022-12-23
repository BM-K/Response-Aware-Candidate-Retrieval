CUDA_VISIBLE_DEVICES=0 python main.py --train True --test False --retrieval_mode base --ckpt retrieval_racar.pt
CUDA_VISIBLE_DEVICES=0 python main.py --train False --test True --retrieval_mode base --ckpt retrieval_racar.pt

CUDA_VISIBLE_DEVICES=0 python main.py --train False --test True --retrieval_mode base --ckpt retrieval_racar.pt --test_data labeled_train.tsv
CUDA_VISIBLE_DEVICES=0 python main.py --train False --test True --retrieval_mode base --ckpt retrieval_racar.pt --test_data labeled_dev.tsv --embedding_mask 656332

mv data/base_labeled_train.tsv ../REGE/data/
mv data/base_labeled_test.tsv ../REGE/data/
mv data/base_labeled_dev.tsv ../REGE/data/
