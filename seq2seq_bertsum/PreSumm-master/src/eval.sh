# export CUDA_VISIBLE_DEVICES=2,3
python train.py -task abs -mode validate -test_all -batch_size 3000 -test_batch_size 500 -bert_data_path ../bert_data -log_file ../logs/val_abs_bert -sep_optim true -use_interval true -alpha 0.95 -result_path ../results/val_abs_bert 

