from __future__ import absolute_import, division, print_function
import sys
sys.path.append("./src")
sys.path.append('..')
from my_transformers import *
import argparse
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
import torch.utils.data.distributed
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import numpy as np
import torch
import torch.nn.init as init
# from run_latent_generation import sample_sequence_conditional
from nltk.translate.bleu_score import corpus_bleu
from transformers import AdamW  # ,OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import GPT2LMHeadModel as GPT2_
from modules import VAE, DDPM, LinearModel, MLPSkipNet, UNetModel,DenseEmbedder, sample_sequence_conditional, TransformerNet
from utils import (calc_iwnll, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split,
                   TextDataset_2Tokenizers, frange_cycle_zero_linear, BucketingMultipleFiles_DataLoader, MultipleFiles_DataLoader)

from train_ddpm_latent import calc_ppl_lgy_ddpm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pudb.remote import set_trace
import shutil, sys


class VAE_DDPM(nn.Module):
    def __init__(self, model_vae, ddpm) :
        super(VAE_DDPM, self).__init__()
        self.model_vae = model_vae
        self.ddpm = ddpm

    def forward(self,inputs, labels, std=False, return_z=False, return_mu=False): 
        
        loss_rec, loss_kl, loss, latent_z, mu = self.model_vae(inputs, labels, std=std, return_z=return_z, return_mu=return_mu)
        ddpm_loss, loss_weight = self.ddpm.forward_new(latent_z, mu)
        
        if self.model_vae.args.ddpm_weight > 0:
            loss = (1/(loss_weight * self.model_vae.args.nt)  * loss).mean() + self.model_vae.args.ddpm_weight *ddpm_loss.mean()
        else:
            loss = loss.mean() + 0.0* ddpm_loss.mean()
        # loss = (1/(loss_weight * self.model_vae.args.nt)  * loss).mean() + self.model_vae.args.ddpm_weight *ddpm_loss.mean()
        return loss_rec, loss_kl, loss, latent_z, mu, ddpm_loss, loss_weight

"""
weights_init_rondom in the original file, changed to random here
"""
def weights_init_random(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)


def collate(examples):
    # Convert to Tensors and build dataset

    input_ids_bert = pad_sequence([torch.tensor(f['bert_token'], dtype=torch.long) for f in examples],
                                  batch_first=True, padding_value=bert_pad_token)
    input_ids_gpt = pad_sequence([torch.tensor(f['gpt2_token'], dtype=torch.long) for f in examples],
                                 batch_first=True, padding_value=gpt2_pad_token)
    try:
        token_lengths = torch.tensor([[len(f['bert_token']), len(f['gpt2_token'])] for f in examples],
                                     dtype=torch.long)
    except:
        token_lengths = torch.zeros((len(examples), 1091))
        for i in range(len(examples)):
            token_lengths[i, len(examples[i]['gpt2_token'])] = 1
    return (input_ids_bert, input_ids_gpt, token_lengths)
logger = logging.getLogger(__name__)


def calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, eval_dataloader,ns=1):
    from modules import sample_sequence_conditional
    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    count = 0
    result = defaultdict(str)
    ref = []
    cand = []
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction", disable=args.disable_bar):
        x0, x1, x_lengths = batch
        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:, :max_len_values[0]]
        x1 = x1[:, :max_len_values[1]]
        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)
        context_tokens = decoder_tokenizer.encode('<BOS>')
        with torch.no_grad():
            # text_x0 = encoder_tokenizer.decode(x0[0,:x_lengths[0,0]].tolist(), clean_up_tokenization_spaces=True)[0]
            # result["INPUT TEXT " + str(count)].append(text_x0)
            attention_mask = (x0 != encoder_tokenizer.pad_token_id).float()

            pooled_hidden_fea = model_vae.encoder(x0, attention_mask)[1]

            # Connect hidden feature to the latent space
            # latent_z, loss_kl = model_vae.connect(pooled_hidden_fea)
            mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            # latent_z = model_vae.reparameterize(mean, logvar, nsamples=1).squeeze(1)
            latent_z = mean.squeeze(1)

            past = latent_z
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=past,
                length=x_lengths[0, 1],  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=latent_z.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id
            )

        for i in range(latent_z.size(0)):
            text_x0_ = decoder_tokenizer.decode(x1[i, :].tolist(), clean_up_tokenization_spaces=False).split('<EOS>')[
                0].replace('<BOS>', '').strip()
            text_x0_ = text_x0_.split()
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split('<EOS>')[
                0].replace('<BOS>', '').strip()
            text_x1 = text_x1.split()

            count += 1
            ref.append([text_x0_])
            cand.append(text_x1)

        if count > 1000:
            break
    bleu = corpus_bleu(ref, cand) * 100
    logger.info("  BLEU = %s", str(round(bleu, 2)))
    output_eval_file = os.path.join(args.output_dir, "eval_results_bleu.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_eval_file, "w") as writer:
        writer.write("%s = %s\n" % ('bleu', str(bleu)))
    return {'bleu': bleu,'ref':ref,'cand':cand}


def calc_ppl_lgy_ddpm(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1, ddpm=None, model=None, tokenizer=None,
                 z=None, total_step=None):
    generate_text = []
    bz = 100
    num_epoch = ns
    context_tokens = decoder_tokenizer.encode(decoder_tokenizer.bos_token)
    def out_(zz):
        generate_text1 = []
        context_tokens = decoder_tokenizer.encode(decoder_tokenizer.bos_token)
        with torch.no_grad():
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=zz,
                length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=zz.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id,
            )
        for i in range(zz.size(0)):
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
            text_x1 = ' '.join(text_x1.split())
            generate_text1.append(text_x1 + '\n')
        return generate_text1

    def text2latent(text):
        # tokenized_text0 = self.encoder_tokenizer.convert_tokens_to_ids(self.encoder_tokenizer.tokenize(text))
        # tokenized_text0 = self.encoder_tokenizer.add_special_tokens_single_sentence(tokenized_text0)
        tokenized_text0 = encoder_tokenizer.encode(text)
        inputs_new = torch.tensor([tokenized_text0]).to(args.device)
        bert_fea = model_vae.encoder(inputs_new)[1]
        mu, _ = model_vae.encoder.linear(bert_fea).chunk(2, -1)
        return mu
    def latent2text(new_z_k):
        out = sample_sequence_conditional(
            model=model_vae.decoder,
            context=context_tokens,
            past=new_z_k.detach(),
            length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
            num_samples=new_z_k.size(0),
            device=args.device,
            decoder_tokenizer=decoder_tokenizer,
            eos_id=model_vae.eos_token_id
        )
        text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
        text_x1 = text_x1.split()
        text_x1 = ' '.join(text_x1)
        print(text_x1)
        return text_x1

    def noise_denoise(text, score_flag = 2):
        latent_z = text2latent(text)
        noisy_z = ddpm.add_noise(latent_z)
        latent_z1 = ddpm.sample_posterior(noisy_z, args.device, score_flag=score_flag)
        out_text = out_(latent_z1)
        return out_text
    loss_list = []
    for _ in trange(num_epoch, desc="Evaluating PPL", disable=True):
        print(_)
        with torch.no_grad():
            latent_z = ddpm.sample_one(bz,(args.latent_size,), args.device,score_flag=2, step=total_step)
            # latent_z = ddpm.sample_new(bz,(args.latent_size,), args.device)
            # import ipdb
            # ipdb.set_trace()
            # latent_z = ddpm.sample_one(bz,(args.latent_size,), args.device, score_flag=2, fp16=args.fp16)
            # latent_z = 0.7 * torch.randn( (bz,args.latent_size )).cuda()
            # text = 'it is very good !'
            # latent_zz = text2latent(text)
            # noisy_z = ddpm.add_noise(latent_zz)
            # latent_z1 = ddpm.sample_posterior(noisy_z, args.device, score_flag=2)
            # out_text = out_(latent_z1[0,0])
            # import pdb
            # pdb.set_trace()
            # text = noise_denoise('unfortunately the system is never coming back .\n', 2)

            # latent_z = gan.generate_z(bz, eval=True)
            loss = True
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=latent_z,
                length=32,
                num_samples=latent_z.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=decoder_tokenizer.eos_token_id,
                loss=loss
            )
            if loss:
                import numpy as np
                loss_ = round(-np.mean(out[1]),3)
                loss_list.append(loss_)
                out = out[0]
        for i in range(latent_z.size(0)):
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split(decoder_tokenizer.eos_token)[
                0].replace(decoder_tokenizer.bos_token, '').strip()
            text_x1 = ' '.join(text_x1.split())
            generate_text.append(text_x1 + '\n')
    # loss_mean = np.mean(loss_list)
    # loss_var = np.var(loss_list)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_text_file = os.path.join(args.output_dir, "out_gene.txt")
    with open(output_text_file,'w') as f:
        f.write(''.join(generate_text))
    encodings = tokenizer('\n\n'.join(generate_text), return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    list_of_references = []
    len_list = []
    for jj, line in enumerate(generate_text):
        if jj < 10:
            if jj == 0: 
                print('\n\n')
            print(line)
        split = line.strip().split(' ')
        list_of_references.append(split)
        len_list.append(len(split))
    # dist1,dist2 = distinct(generate_text)
    # score  = 10*(-dist2-dist1)
    sbleu = []
    num_all = len(list_of_references)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    for i in range(num_all):
        refs = [list_of_references[j] for j in range(num_all) if i != j]
        bleu_ = sentence_bleu(refs, list_of_references[i], smoothing_function=SmoothingFunction().method1)
        sbleu.append(bleu_ * 100)
    score = np.mean(sbleu)
    # weights = {'4gram': (1 / 4., 1 / 4., 1 / 4., 1 / 4.)}
    # from fast_bleu import SelfBLEU
    # self_bleu = SelfBLEU(list_of_references, weights)
    # score = np.mean(self_bleu.get_score()['4gram']) * 100
    len_mean = np.mean(len_list)
    norm_z = latent_z.norm(dim=-1).mean().item()
    return {'ppl': ppl, 'sbleu': round(score, 2), 'length': round(len_mean, 2), 'norm_z': norm_z,
            'ppl_sbleu': ppl + round(score, 2),'generate_text':generate_text}

MODEL_CLASSES = {
    'gpt2': GPT2ForLatentConnectorNew, # updated by Jieqi, for 84 only
    # 'openai-gpt': (None, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bertu': BertForLatentConnectorAVG, # updated by Jiei, for 84 only
    'bert': BertForLatentConnector,
    'roberta': RobertaForLatentConnector,
    'deberta': DebertaForLatentConnector,
    't5': T5EncoderForLatentConnector,
    'albert':AlbertForLatentConnector,
}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default=None, type=str, help="The dataset.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--ExpName", default="", type=str,
                        help="The experiment name used in Azure Table.")
    parser.add_argument("--save_bert_gpt_init", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--length_weighted_loss", action='store_true',
                        help="Use sentence length re-weight the reconstruction loss.")

    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_pretrained_model", action='store_true',
                        help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument("--latent_as_gpt_memory", default=1, type=int,
                        help="Latent vector as memery for GPT2 to attend.")
    parser.add_argument("--latent_as_gpt_emb", default=1, type=int, help="Latent vector as embeddings for GPT2.")

    ## Objective functions
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="The weighting hyper-parameter of the KL term in VAE")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")
    parser.add_argument("--block_size", default=30, type=int,
                        help="Optional input sequence length after tokenization."
                            "The training dataset will be truncated in block of this size for training."
                            "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_rec", action='store_true',
                        help="Whether to run eval reconstruction on a set of models.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Training Schedules
    parser.add_argument("--ratio_increase", default=0.1, type=float,
                        help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.5, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--fb_mode", default=1, type=int,
                        help="free bit training mode.")
    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--use_pretrained_vae", action='store_true',
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--use_random_weight", action='store_true',
                        help="Use random weights as initialization")

    ## IO: Logging and Saving
    parser.add_argument('--logging_steps', type=int, default=-1,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gloabl_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    # Precision & Distributed Training 
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    ############## Mine
    parser.add_argument('--fix_model', type=int, default=84,
                        help="0: no fix; 1: fix both bert & gpt; 2: fix gpt; 3: fix both bert & gpt, extra layers")
    parser.add_argument('--disable_bar', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--nt', type=int, default=1000, help="T for diffusion process")
    parser.add_argument('--shell_name', type=str, default='', help="shell name")
    parser.add_argument("--ddpm_pretrain", type=int, default=0,
                        help="Use pretrained DDPM")
    parser.add_argument('--ddpm_weight', type=float, default=1.0)
    # args = parser.parse_args()
    # args.n_gpu = torch.cuda.device_count()
    # torch.distributed.init_process_group(backend='nccl',init_method='env://')
    # torch.cuda.set_device(args.local_rank)
    # device = torch.device('cuda', args.local_rank)
    args = parser.parse_args()
    args.device = "cuda"

    #encoder
    encoder_model_class = MODEL_CLASSES[args.encoder_model_type]
        # encoder_config = encoder_config_class.from_pretrained(
        #     args.encoder_config_name if args.encoder_config_name else args.encoder_model_name_or_path)
    tokenizer_encoder = AutoTokenizer.from_pretrained(
        args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path,
        do_lower_case=args.do_lower_case, local_files_only=False)
    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path, latent_size=args.latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id,local_files_only=False)

        
    ## Decoder
    decoder_model_class = MODEL_CLASSES[args.decoder_model_type]
    tokenizer_decoder = AutoTokenizer.from_pretrained(
        args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path,
        do_lower_case=args.do_lower_case, local_files_only=False)
    if args.block_size <= 0:
        args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
    latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False    
    model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, latent_size=args.latent_size,
                                                    latent_as_gpt_emb=latent_as_gpt_emb,
                                                    latent_as_gpt_memory=latent_as_gpt_memory,local_files_only=False)

    decoder_n_layer = model_decoder.transformer.config.n_layer
    if args.fix_model == 3 or args.fix_model == 4:
        print("Initialize the Extra Layer.")
        model_encoder.linear_forbert.load_state_dict(model_encoder.encoder.layer[-1].state_dict())
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.change_order()
        print('Change the Order of Decoder Layers')
    elif args.fix_model == 5:
        print("Initialize the Extra Layer.")
        model_encoder.linear_forbert.load_state_dict(model_encoder.encoder.layer[0].state_dict())
    elif args.fix_model == 6 or args.fix_model == 8 or args.fix_model == 85 or args.fix_model == 881  or args.fix_model == 882 or args.fix_model == 883:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.change_order()
    elif args.fix_model == 84:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.change_order()
    elif args.fix_model == 10 or args.fix_model == 11:
        from transformers.adapters import CompacterConfig
        config = CompacterConfig(reduction_factor=4)
        model_decoder.transformer.add_adapter("dummy", config=config)
        model_decoder.transformer.train_adapter("dummy")
        # model_decoder.transformer.train_adapter("poem")
    elif args.fix_model == 12:
        aa = model_decoder.transformer.load_adapter("/home/guangyiliu/yiwen_Optimus/output/adapters", model_name='gpt2')
        model_decoder.transformer.train_adapter(aa)
    elif args.fix_model == 13 or args.fix_model == 14 or args.fix_model == 82:
        model_decoder.transformer.h[decoder_n_layer+1].load_state_dict(model_decoder.transformer.h[0].state_dict())
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[11].state_dict())
        model_decoder.transformer.change_order(extra_num=2)
    elif args.fix_model == 83:
        model_decoder.transformer.h[decoder_n_layer].load_state_dict(model_decoder.transformer.h[11].state_dict())
        model_decoder.transformer.config.n_layer += 1

    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    global bert_pad_token, gpt2_pad_token
    bert_pad_token = tokenizer_encoder.pad_token_id
    gpt2_pad_token = tokenizer_decoder.pad_token_id
    model_decoder.resize_token_embeddings(
            len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'
    tokenizer_decoder.convert_tokens_to_ids([tokenizer_decoder.pad_token])[0]
    checkpoint_full_path = os.path.join(args.checkpoint_dir,"checkpoint-full-2/training.bin")
    checkpoint_full = torch.load(checkpoint_full_path,map_location=torch.device('cuda', args.local_rank))

    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)
    model_vae.load_state_dict(checkpoint_full['model_state_dict'], strict=False)
    ddpm = DDPM(eps_model=MLPSkipNet(args.latent_size), betas=(1e-4, 0.02), n_T=args.nt, criterion=nn.MSELoss(reduction='none'),)
    checkpoint_full_path = os.path.join(args.checkpoint_dir,"checkpoint-ddpm-2-1/training_ddpm.bin")
    ddpm_checkpoint = torch.load(checkpoint_full_path,map_location=torch.device('cuda', args.local_rank))
    ddpm.apply(weights_init_random)
    ddpm.load_state_dict(ddpm_checkpoint['model_state_dict'], strict=False)
    
    ddpm.to(args.device)
    model_vae.to(args.device)
    print('moved to cuda')
    model = VAE_DDPM(model_vae, ddpm).cuda()
    model_id ='gpt2'
    if args.dataset == 'yelp':
        model_id =  '../classifiers/gpt2_yelp'

    model_ppl = GPT2_.from_pretrained(model_id,local_files_only=False).cuda()
    tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id,local_files_only=False)   
    train_eval_datasets=load_dataset(args.train_data_file)
    eval_dataloader =  DataLoader(train_eval_datasets['test'], num_workers=0, collate_fn=collate,batch_size=args.per_gpu_eval_batch_size)
    table_name = 'Vae' + args.dataset + 'Nz' + str(args.latent_size)
    args.n_gpu = 1
    ###generation
    generation_result= calc_ppl_lgy_ddpm(model.model_vae, tokenizer_encoder, tokenizer_decoder, args, 1,model.ddpm, model_ppl, tokenizer_ppl, z=None)
    ###reconstruction
    cal_rec_lgy_result = calc_rec_lgy(model_vae, tokenizer_encoder, tokenizer_decoder, args, eval_dataloader,ns=100)

    return None


if __name__ == "__main__":
    main()


