# export PYTHONPATH="${PYTHONPATH}:/workspace/code"
# export TRANSFORMERS_CACHE='/home/liping.tang'
# export MASTER_PORT=29501

data=yelp
dataset=$data

TRAIN_FILE=guangyil/yelp_short_v2 #amazon_tokenized #SBU_caption #yelp_short_v2
TEST_FILE=guangyil/yelp_short_v2 #amazon_tokenized #SBU_caption #yelp_short_v2



beta=0.0
latent_size=128
epoch=20 #50
ratio_zero=0.01
learning_rate=9e-5 #1e-4 #5e-5 2.5e-5
ratio_increase=0.1
batch=32 #64
eval_batch=64 #32

apex_opt=O2
fix_model=84

cuda=0
ddpm_weight=10

args=" --use_pretrained_vae --use_pretrained_model  $2  " #--disable_bar --no_save --disable_bar --disable_bar --use_pretrained_vae --use_pretrained_model --use_pretrained_vae --use_pretrained_model

model='bertu'

model_path='prajjwal1/bert-small' #'../output/bert-base-uncased' #bert-small' bert-base-uncased

gpt_path=gpt2-xl
num_gpu=1
sym='ft_T1000_w'$ddpm_weight
echo $sym
ckpt='../ckpts/Apr9_ftyelp_2gpu_128_b128_e50_b0.0_lr5e-5_w8' #Apr8_nopre_book_w5_128_b128_e5_b0.0_lr9e-5'  # book/Apr8_nopre_book_w5_128_b128_e5_b0.0_lr9e-5'  #/Mar21_4_new_w5_128_b128_e50_b0.0_lr5e-5   --master_port=29501 
name='Apr17_'$sym'_'$latent_size'_b'$batch'_e'$epoch'_b'$beta'_lr'$learning_rate'_w'$ddpm_weight
#torch.distributed.launch --nproc_per_node=$num_gpu --master_port=29503
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=$num_gpu --master_port=29503 train.py \
    --output_dir=/data/yunhao/out/LM/$data/$name  \
    --dataset $dataset \
    --encoder_model_type=$model \
    --encoder_model_name_or_path=$model_path \
    --decoder_model_type=gpt2 \
    --decoder_model_name_or_path=$gpt_path \
    --beta $beta \
    --do_train \
    --do_eval \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs $epoch \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=$batch \
    --per_gpu_eval_batch_size=$eval_batch \
    --block_size 32 \
    --length_weighted_loss \
    --latent_size $latent_size \
    --evaluate_during_training \
    --gloabl_step_eval 1 --ddpm_pretrain 1 \
    --checkpoint_dir $ckpt \
    --learning_rate $learning_rate --fix_model $fix_model    --shell_name ${0##*/}  --nt 2000 --ddpm_weight $ddpm_weight \
    --fp16_opt_level  $apex_opt --fp16 $args #  2>&1|tee out/$name.out 



# ddpm_weight=5

# args=" $2 --use_pretrained_vae --use_pretrained_model " #--disable_bar --no_save --disable_bar --disable_bar --use_pretrained_vae --use_pretrained_model --use_pretrained_vae --use_pretrained_model

# model='bertu'

# model_path='prajjwal1/bert-small' #'../output/bert-base-uncased' #bert-small' bert-base-uncased

# gpt_path=gpt2-xl
# num_gpu=4
# sym='final_64_w'$ddpm_weight
# echo $sym
# ckpt='../output_home/LM/book/Apr13_final_64_w5_64_b128_e5_b0.0_lr2e-4' #Apr8_nopre_book_w5_128_b128_e5_b0.0_lr9e-5'  # book/Apr8_nopre_book_w5_128_b128_e5_b0.0_lr9e-5'  #/Mar21_4_new_w5_128_b128_e50_b0.0_lr5e-5   --master_port=29501 
# name='Apr13_'$sym'_'$latent_size'_b'$batch'_e'$epoch'_b'$beta'_lr'$learning_rate'_w'$ddpm_weight
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpu --master_port=29501  examples/big_ae/run_lm_joint_vae_training_pretraining_new_DDP.py \
#     --output_dir=../output_home/LM/$data/$name  \
#     --dataset $dataset \
#     --encoder_model_type=$model \
#     --encoder_model_name_or_path=$model_path \
#     --decoder_model_type=gpt2 \
#     --decoder_model_name_or_path=$gpt_path \
#     --beta $beta \
#     --do_train \
#     --do_eval \
#     --train_data_file=$TRAIN_FILE \
#     --eval_data_file=$TEST_FILE \
#     --num_train_epochs $epoch \
#     --overwrite_output_dir \
#     --per_gpu_train_batch_size=$batch \
#     --per_gpu_eval_batch_size=$eval_batch \
#     --block_size 32 \
#     --length_weighted_loss \
#     --latent_size $latent_size \
#     --evaluate_during_training \
#     --gloabl_step_eval 1 --ddpm_pretrain 1 \
#     --checkpoint_dir $ckpt \
#     --learning_rate $learning_rate --fix_model $fix_model    --shell_name ${0##*/}  --nt 2000 --ddpm_weight $ddpm_weight \
#     --fp16_opt_level  $apex_opt --fp16 $args #  2>&1|tee out/$name.out 
