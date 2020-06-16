export FLAGS_fraction_of_gpu_memory_to_use=0.7
python train.py -task abs -mode train -bert_data_path ../bert_data -load_from_extractive ../models/bert_ext/model_step_50000.pt -dec_dropout 0.2 -model_path ../models/bert_ext_abs -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 140 -train_steps 200000 -report_every 50 -accum_count 10 -use_bert_emb true -use_interval true -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2 -log_file ../logs/bert_abs -train_from ../models/bert_ext_abs/model_step_92000.pt 

