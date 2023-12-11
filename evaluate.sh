python -m evaluate \
    --output_dir /data/yunhao/out/experiment\
    --dataset yelp  \
    --encoder_model_type    bertu  \
    --encoder_model_name_or_path    prajjwal1/bert-small  \
    --decoder_model_type    gpt2  \
    --decoder_model_name_or_path  gpt2-xl \
    --beta 0.0 --do_eval \
    --local_rank 0  \
    --train_data_file guangyil/yelp_short_v2  \
    --eval_data_file guangyil/yelp_short_v2 --num_train_epochs 50  --overwrite_output_dir \
    --per_gpu_train_batch_size 128  \
    --per_gpu_eval_batch_size 2    \
    --block_size    32     \
    --length_weighted_loss  \
    --latent_size    128     \
    --evaluate_during_training --gloabl_step_eval 1   \
    --ddpm_pretrain    1  --checkpoint_dir /data/yunhao/ckpts/Apr9_ftyelp_2gpu_128_b128_e50_b0.0_lr5e-5_w8  \
    --learning_rate 5e-5    --fix_model    84   \
    --nt    2000    \
    --ddpm_weight    8 --fp16_opt_level O2  --fp16  --use_pretrained_vae  --use_pretrained_model