# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from __future__ import absolute_import, division, print_function
import os

import argparse
import logging

import random

import numpy as np
import torch
import torch.nn.init as init
from my_transformers import *
# from pytorch_transformers import AdamW
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AdamW
# from run_latent_generation import sample_sequence_conditional
from transformers import get_polynomial_decay_schedule_with_warmup

from modules import GAN  # GANVAE as GAN
from modules import VAE, DenseEmbedder, sample_sequence_conditional, DDPM, LinearModel, MLPSkipNet, UNetModel
from utils import (BucketingDataLoader, TextDataset_Split,
                   TextDataset_2Tokenizers, frange_cycle_zero_linear)
import time
from tensorboardX import SummaryWriter
from npy_append_array import NpyAppendArray
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': GPT2ForLatentConnector,
    # 'openai-gpt': (None, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': BertForLatentConnector,
    'bertu': BertForLatentConnector,
    'roberta': RobertaForLatentConnector,
    'deberta': DebertaForLatentConnector,
    't5': T5EncoderForLatentConnector,
}

#### GPT2 for ppl
from transformers import GPT2LMHeadModel as GPT2_
from transformers import GPT2TokenizerFast, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader

# model_id = '../output/gpt2_styleptb'  # sentiment'  # _sentiment' #amazon'
# model_ppl = GPT2_.from_pretrained(model_id).cuda()
# tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id)

#### GPT2 for ppl
start_time = time.time()
class LatentDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, latent_z, labels):
        self.latent_z = latent_z
        self.labels = labels

    def __len__(self):
        return len(self.latent_z)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'latent_z': self.latent_z[idx], 'labels': self.labels[idx]}
        return sample


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers(tokenizer, args,
                                          file_path=args.eval_data_file if evaluate else args.train_data_file,
                                          block_size=args.block_size)
    else:
        dataset = TextDataset_Split(tokenizer, args,
                                    file_path=args.eval_data_file if evaluate else args.train_data_file,
                                    block_size=args.block_size)
    return dataset


def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        if not evaluate:
            args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            file_path = args.train_data_file
        else:
            args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            file_path = args.eval_data_file
        dataloader = BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100,
                                         shuffle=False)
    else:
        pass
    return dataloader


def distinct(lines):
    for i, line in enumerate(lines):
        lines[i] = line.strip().split()
    grams = lines
    grams_list1 = []
    for sen in grams:
        for g in sen:
            grams_list1.append(g)

    grams_list2 = []
    for sen in grams:
        for i in range(len(sen) - 1):
            grams_list2.append(str(sen[i]) + ' ' + str(sen[i + 1]))
    dist1 = round(len(set(grams_list1)) / len(grams_list1), 4)
    dist2 = round(len(set(grams_list2)) / len(grams_list2), 4)
    return (dist1, dist2)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).to(torch.uint8)
    labels[masked_indices == 1] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
    indices_random = indices_random
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def weights_init_rondom(model):
    model = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if 'encoder' in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)


def save_cls_checkpoint(classifier, optimizer, global_step, args, gan=None, eval_loss=False):
    # Create output directory if needed
    # Save model checkpoint
    save_last = args.save_step
    save_ddpm_step = args.gloabl_step_eval
    output_cls_dir = os.path.join(args.output_dir, 'checkpoint-cls-{}'.format(save_last))
    if not os.path.exists(output_cls_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_cls_dir)
    if args.ddpm_pretrain:
        save_ddpm_step = str(args.gloabl_step_eval) + '-1'
    output_gan_dir = os.path.join(args.output_dir, 'checkpoint-ddpm-{}'.format(save_ddpm_step))
    if not os.path.exists(output_gan_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_gan_dir)
    logger.info("Saving DDPM model checkpoint to %s", output_gan_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`

    model_gan_to_save = gan.module if hasattr(gan,
                                              'module') else gan  # Take care of distributed/parallel training

    checkpoint_gan = {
        'iter': global_step,
        'model_state_dict': model_gan_to_save.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    torch.save(checkpoint_gan, os.path.join(output_gan_dir, 'training_ddpm.bin'))
    logger.info("Saving DDPM checkpoint to %s", output_gan_dir)



def access_latent_label(args, train_dataloader, model_vae, train=True):
    """ Train the model """
    npy_file_path = args.train_data_file if train else args.eval_data_file
    whole_path = npy_file_path + '_' +args.output_dir.split('/')[-1]+ '_'+str(args.gloabl_step_eval)+'.npy'
    print(whole_path)
    if os.path.exists(whole_path):
        all_data = np.load(whole_path,mmap_mode="r")
        # with open(whole_path, 'rb') as f:
        #     all_data = np.load(f)
            # all_z = all_data[:,:-1]
            # all_label = all_data[:,-1]
    else:
        all_z = np.zeros((0, args.latent_size))
        all_label = np.zeros((0), )
        epoch_iterator = tqdm(train_dataloader, desc="Creating Latent data")
        with NpyAppendArray(whole_path,delete_if_exists=True) as npaa:
            for step, batch in enumerate(epoch_iterator):
                tokenized_text0, tokenized_text1, tokenized_text_lengths = batch
                latent_labels = tokenized_text_lengths[:, -1]

                inputs = tokenized_text0
                inputs = inputs.to(args.device)
                model_vae.eval()
                with torch.no_grad():
                    latent_z = model_vae.encode_x(inputs)
                    zz = np.append(latent_z.cpu().numpy(), latent_labels.numpy()[:, None],1 )
                    npaa.append(zz)
        all_data = np.load(whole_path,mmap_mode="r")
        # all_z = all_data[:,:-1]
        # all_label = all_data[:,-1]
                    # all_z = np.append(all_z, latent_z.cpu().numpy(), 0)
                    # all_label = np.append(all_label, latent_labels.numpy(), 0)
            # all_data = np.append(all_z,all_label[:,None],1)
        # with open(whole_path, 'wb') as f:
        #     np.save(f,all_data)
    return [all_data[:,:-1], all_data[:,-1]]

def access_latent_label_ddim(args, train_dataloader, model_vae, ddpm, train=True):
    """ Train the model """
    npy_file_path = args.train_data_file if train else args.eval_data_file
    whole_path = npy_file_path + '_' +args.output_dir.split('/')[-1]+ 'ddim.npy'
    whole_path_ori = npy_file_path + '_' +args.output_dir.split('/')[-1]+ 'ori.npy'
    print(whole_path)
    ddpm.eval()
    all_z = np.zeros((0, args.latent_size))
    all_label = np.zeros((0), )
    epoch_iterator = tqdm(train_dataloader, desc="Creating Latent data")
    with NpyAppendArray(whole_path,delete_if_exists=True) as npaa:
        with NpyAppendArray(whole_path_ori,delete_if_exists=True) as npaa_ori:
            for step, batch in enumerate(epoch_iterator):
                latent_z = batch['latent_z'].float().to(args.device)
                with torch.no_grad():
                    latent_zt = ddpm.add_noise(latent_z,T=2000)
                    npaa.append(latent_zt.cpu().numpy())
                    npaa_ori.append(latent_z.cpu().numpy())
    all_data = np.load(whole_path,mmap_mode="r")
    # all_z = all_data[:,:-1]
    # all_label = all_data[:,-1]
                # all_z = np.append(all_z, latent_z.cpu().numpy(), 0)
                # all_label = np.append(all_label, latent_labels.numpy(), 0)
        # all_data = np.append(all_z,all_label[:,None],1)
    # with open(whole_path, 'wb') as f:
    #     np.save(f,all_data)
    import ipdb
    ipdb.set_trace()
    return [all_data[:,:-1], all_data[:,-1]]



def train_ddpm(args, train_dataloader, model_vae, encoder_tokenizer, decoder_tokenizer, ddpm, eval_latent_dataset):
    """ Train the ddpm model """
    tb_writer = SummaryWriter('./runs/' + args.output_dir.split('/')[-2] + '/' + args.output_dir.split('/')[-1]+'/ddpm_'+str(time.time())[-5:])
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer_grouped_parameters = [
        {'params': [p for n, p in ddpm.named_parameters()],
         'weight_decay': 0.0},
    ]
    
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps=t_total, lr_end=5e-7, power=3.0)
    if args.fp16:
        from apex import amp
        import apex
        optimizer = apex.optimizers.FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        ddpm, optimizer = amp.initialize(ddpm, optimizer,
                                         opt_level=args.fp16_opt_level)
        model_vae = amp.initialize(model_vae,
                                         opt_level=args.fp16_opt_level)
        # if 'cls' in args.train_cls_gan:
        #     classifier, optimizer = amp.initialize(classifier, optimizer, opt_level=args.fp16_opt_level)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps=t_total, lr_end=5e-7, power=3.0)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model_vae.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=int(args.num_train_epochs),
                                           ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    args.logging_steps = int(np.floor(len(train_dataloader))) // 5
    args.save_steps = args.logging_steps

    stop_flag = False
    best_gan_diff = 200
    best_diff_cnt = 0
    
    # results = calc_ppl_lgy_ddpm(
    #     model_vae, encoder_tokenizer, decoder_tokenizer, args, 1,
    #     ddpm, model_ppl, tokenizer_ppl
    # )
    start_time = time.time()
    train_step = 0
    dtype_ = torch.half if args.fp16 else torch.float
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        if best_gan_diff < 200:
            use_time = time.time() - start_time
            start_time = time.time()
            logger.info("Time for this epoch = %f", use_time)
        for step, batch in enumerate(epoch_iterator):

            # tokenized_text0, _, tokenized_text_lengths = batch
            # latent_labels = tokenized_text_lengths[:, -1]
            mu = batch['latent_z'].float().to(args.device)
            logvar = torch.log(torch.ones_like(mu) * 0.008)
            latent_z = model_vae.reparameterize(mu, logvar, nsamples=1).squeeze(1)
            # latent_labels = batch['labels'].to(args.device)
            model_vae.eval()
            ddpm.train()

            # loss = ddpm(latent_z)
            loss, _ = ddpm.forward_new(latent_z.to(dtype_), mu.to(dtype_)) # 64 , loss_weight
            loss = loss.mean()
            if train_step % 100 == 0:
                tb_writer.add_scalar('ddpm_loss_train', loss.mean().item(), train_step)
                tb_writer.add_scalar('ddpm_lr_train', scheduler.get_last_lr()[0], train_step)
            train_step += 1
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(ddpm.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                ddpm.zero_grad()

                epoch_iterator.set_description(
                    (
                        f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                        f'loss_d: {loss.item():.3f};'
                    )
                )
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    ddpm.eval()
                    results = calc_ppl_lgy_ddpm(
                        model_vae, encoder_tokenizer, decoder_tokenizer, args, 1, 
                        ddpm, model_ppl,tokenizer_ppl, z=latent_z
                    )

                    logger.info("PPL = %f", results['ppl'])
                    logger.info("sBLEU = %f", results['sbleu'])
                    logger.info("PPL+sBLEU = %f", results['ppl_sbleu'])
                    logger.info("Length = %f", results['length'])
                    logger.info("z norm = %f", results['norm_z'])
                    tb_writer.add_scalar('ddpm_ppl', results['ppl'], train_step)
                    tb_writer.add_scalar('ddpm_sBLEU', results['sbleu'], train_step)
                    tb_writer.add_scalar('ddpm_PPL_sBLEU', results['ppl_sbleu'], train_step)
                    if results['ppl'] < best_gan_diff and results['ppl'] > 10 and results['norm_z']<15:
                        best_gan_diff = results['ppl']
                        best_diff_cnt = 0
                        save_cls_checkpoint(None, optimizer, global_step, args, gan=ddpm)
                        tb_writer.add_scalar('eval_best_ppl', results['ppl'], train_step)
                        tb_writer.add_scalar('eval_best_ppl_sbleu', results['ppl_sbleu'], train_step)
                        tb_writer.add_scalar('eval_best_sbleu', results['sbleu'], train_step)
                    else:
                        best_diff_cnt += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return 0



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
        with torch.no_grad():
            # latent_z = ddpm.sample_one(bz,(args.latent_size,), args.device,score_flag=1, step=total_step)
            latent_z = ddpm.sample_new(bz,(args.latent_size,), args.device, fp16=args.fp16)
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
    with open('out_gene.txt','w') as f:
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
            'ppl_sbleu': ppl + round(score, 2)}



def calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args, ns=1):
    from nltk.translate.bleu_score import corpus_bleu
    eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    count = 0
    ref = []
    cand = []
    def out_(zz):
        generate_text1 = []
        context_tokens = decoder_tokenizer.encode('<BOS>')
        with torch.no_grad():
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=zz,
                length=20,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
                num_samples=zz.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id
            )
        for i in range(zz.size(0)):
            text_x1 = decoder_tokenizer.decode(out[i, :].tolist(), clean_up_tokenization_spaces=False).split('<EOS>')[
                0].replace('<BOS>', '').strip()
            text_x1 = ' '.join(text_x1.split())
            generate_text1.append(text_x1 + '\n')
        return generate_text1

    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction"):
        x0, x1, x_lengths = batch
        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        # x_lengths = x_lengths.to(args.device)
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
                length=30,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
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
    return {'bleu': bleu}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters '/home/guangyi/LatentOps/data/datasets/yelp_data/train.shuf.merge'  '/home/guangyi/guangyi/bookcorpus/json/bookcorpus_split_0.txt'
    parser.add_argument("--train_data_file", default='/home/guangyi.liu/LatentOps/data/datasets/yelp_data/train.shuf.merge', type=str,
                        help="The input training data file (a text file).")
                        #  ../output_home/LM/yelp/new_VAE_128_b64_e10_b0.9_lr3e-5   v2_new_VAE_128_b64_e40_b3.0_lr4e-5 # ../output_home/LM/caption/Feb8_caption_VAE_128_b128_e40_b0.9_lr5e-5_bertu_dkl0.9
    parser.add_argument("--checkpoint_dir", default='../output_home/LM/yelp/Apr9_ftyelp_2gpu_128_b128_e50_b0.0_lr5e-5_w8', type=str,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--output_dir", default='../output_home/LM/book/Apr8_nopre_book_w5_128_b128_e5_b0.0_lr9e-5', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default='Yelp_cls', type=str, help="The dataset.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default='../data/datasets/yelp_data/test.merge', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--ExpName", default="", type=str,
                        help="The experiment name used in Azure Table.")
    parser.add_argument("--save_bert_gpt_init", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--length_weighted_loss", default=1, type=int,
                        help="Use sentence length re-weight the reconstruction loss.")

    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bertu", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="prajjwal1/bert-small", type=str,  #
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="gpt2-xl", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    ## Variational auto-encoder
    parser.add_argument("--latent_size", default=128, type=int, help="Latent space dimension.")
    parser.add_argument("--use_deterministic_connect", action='store_true',
                        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.")
    parser.add_argument("--use_pretrained_model", type=int, default= 1.0,
                        help="Use pre-trained auto-encoder models as the initialization")
    parser.add_argument("--latent_as_gpt_memory", default=1, type=int,
                        help="Latent vector as memery for GPT2 to attend.")
    parser.add_argument("--latent_as_gpt_emb", default=1, type=int, help="Latent vector as embeddings for GPT2.")

    ## Objective functions
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="The weighting hyper-parameter of the KL term in VAE")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")
    parser.add_argument("--block_size", default=30, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", default=1, type=int,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=1, type=int,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_rec", action='store_true',
                        help="Whether to run eval reconstruction on a set of models.")
    parser.add_argument("--evaluate_during_training", default=1, type=int,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Training Schedules
    parser.add_argument("--ratio_increase", default=0.25, type=float,
                        help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.25, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")
    parser.add_argument("--fb_mode", default=5, type=int,
                        help="free bit training mode.")
    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")
    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int,  # 256
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,  # 32  
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=4e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument("--use_pretrained_vae", type=int, default= 1.0,
                        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder")
    parser.add_argument("--use_random_weight", action='store_true',
                        help="Use random weights as initialization")

    ## IO: Logging and Saving
    parser.add_argument('--logging_steps', type=float, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=898,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default=1, type=int,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gloabl_step_eval', type=int, default=1,
                        help="Evaluate the results at the given global step")

    # Precision & Distributed Training
    parser.add_argument('--fp16', type=int,default=1,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--train_cls_gan', type=str, default='cls')
    parser.add_argument('--n_cyc', type=int, default=5)
    parser.add_argument('--save_step', type=str, default=1)
    parser.add_argument('--fix_model', type=int, default=84,
                        help="0: no fix; 1: fix both bert & gpt; 2: fix gpt; 3: fix both bert & gpt, extra layers")
    parser.add_argument('--nt', type=int, default=2000, help="T for diffusion process")
    parser.add_argument('--ddpm_pretrain',type=int, default=1)
    args = parser.parse_args()
    args.output_dir = args.checkpoint_dir
    if 'book' in args.train_data_file:
        model_id = 'gpt2'
    else:
        model_id ='../classifiers/gpt2_yelp' # + args.output_dir.split('/')[-1]  # sentiment'  # _sentiment' #amazon'
    print(model_id)
    global model_ppl
    model_ppl = GPT2_.from_pretrained(model_id).cuda()
    global tokenizer_ppl
    tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id)
    MODEL_CLASSES['bertu'] = BertForLatentConnectorAVG 
    MODEL_CLASSES['bert'] = BertForLatentConnectorAVG
    if 'large' in args.decoder_model_name_or_path or 'xl' in args.decoder_model_name_or_path:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew
    else:
        MODEL_CLASSES['gpt2'] = GPT2ForLatentConnectorNew2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    global_step = args.gloabl_step_eval
    output_full_dir = os.path.join(args.checkpoint_dir, 'checkpoint-full-{}'.format(global_step))

    checkpoint = torch.load(os.path.join(output_full_dir, 'training.bin'))

    ## Encoder
    encoder_model_class = MODEL_CLASSES[args.encoder_model_type]
    tokenizer_encoder = AutoTokenizer.from_pretrained(
        args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path,
        do_lower_case=args.do_lower_case)

    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    model_encoder = encoder_model_class.from_pretrained(args.encoder_model_name_or_path, latent_size=args.latent_size,
                                                        pad_id=tokenizer_encoder.pad_token_id)

    ## Decoder
    decoder_model_class = MODEL_CLASSES[args.decoder_model_type]
    tokenizer_decoder = AutoTokenizer.from_pretrained(
        args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path,
        do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
    latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False

    # setattr(decoder_config, "latent_size", args.latent_size)
    model_decoder = decoder_model_class.from_pretrained(args.decoder_model_name_or_path, latent_size=args.latent_size,
                                                        latent_as_gpt_emb=latent_as_gpt_emb,
                                                        latent_as_gpt_memory=latent_as_gpt_memory)
    model_decoder.transformer.change_order()

    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    model_decoder.resize_token_embeddings(len(tokenizer_decoder)) 
    model_vae = VAE(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args)

    model_vae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #

    ddpm = DDPM(eps_model=MLPSkipNet(args.latent_size), betas=(1e-4, 0.02), n_T=args.nt, criterion=nn.MSELoss(reduction='none'))
    # ddpm = DDPM(eps_model=UNetModel(), betas=(1e-4, 0.02), n_T=args.nt,)
    # ddpm = DDPM(eps_model=LinearModel(args.latent_size), betas=(1e-4, 0.02), n_T=args.nt, )
    ddpm.to(args.device)
    ddpm.apply(weights_init_rondom)
    if args.ddpm_pretrain or (not args.do_train):
        ddpm_ckpt = torch.load(os.path.join(args.output_dir, 'checkpoint-ddpm-{}/training_ddpm.bin'.format(global_step)))
        ddpm.load_state_dict(ddpm_ckpt['model_state_dict'], strict=False)
    import copy
    args_ = copy.deepcopy(args)
    args_.per_gpu_train_batch_size = args.per_gpu_train_batch_size * 8
    

    if args.do_train:
        train_dataloader = build_dataload_and_cache_examples(args_, [tokenizer_encoder, tokenizer_decoder],
                                                         evaluate=False)
        all_z, all_label = access_latent_label(args, train_dataloader, model_vae, train=True)
        latent_dataset = LatentDataset(all_z, all_label)

        dataloader = DataLoader(latent_dataset, batch_size=args.per_gpu_train_batch_size,
                                shuffle=True, num_workers=0)
        train_ddpm(args, dataloader, model_vae, tokenizer_encoder, tokenizer_decoder, ddpm=ddpm,
              eval_latent_dataset=None)
    
    if not args.do_train and args.do_eval:
        eval_dataloader = build_dataload_and_cache_examples(args_, [tokenizer_encoder, tokenizer_decoder],
                                                            evaluate=True)
        # eval_z, eval_label = access_latent_label(args, eval_dataloader, model_vae, train=False)
        # eval_latent_dataset = LatentDataset(eval_z, eval_label)
        # dataloader = DataLoader(eval_latent_dataset, batch_size=args.per_gpu_train_batch_size,
        #                         shuffle=False, num_workers=0)
        # _ ,_ = access_latent_label_ddim(args, dataloader, model_vae, ddpm, train=False)
        args.fp16=0
        results = calc_ppl_lgy_ddpm(
            model_vae, tokenizer_encoder, tokenizer_decoder, args, 1,
            ddpm, model_ppl, tokenizer_ppl, z=None, total_step=2000
        )
        logger.info("PPL = %f", results['ppl'])
        logger.info("sBLEU = %f", results['sbleu'])
        logger.info("PPL+sBLEU = %f", results['ppl_sbleu'])
        logger.info("Length = %f", results['length'])
        logger.info("z norm = %f", results['norm_z'])
        # results = calc_ppl_lgy_ddpm(
        #     model_vae, tokenizer_encoder, tokenizer_decoder, args, 1,
        #     ddpm, model_ppl, tokenizer_ppl, z=None, total_step=500
        # )
        # logger.info("PPL = %f", results['ppl'])
        # logger.info("sBLEU = %f", results['sbleu'])
        # results = calc_ppl_lgy_ddpm(
        #     model_vae, tokenizer_encoder, tokenizer_decoder, args, 1,
        #     ddpm, model_ppl, tokenizer_ppl, z=None, total_step=200
        # )
        # logger.info("PPL = %f", results['ppl'])
        # logger.info("sBLEU = %f", results['sbleu'])
        # results = calc_ppl_lgy_ddpm(
        #     model_vae, tokenizer_encoder, tokenizer_decoder, args, 1,
        #     ddpm, model_ppl, tokenizer_ppl, z=None, total_step=100
        # )
        # logger.info("PPL = %f", results['ppl'])
        # logger.info("sBLEU = %f", results['sbleu'])
        # results = calc_ppl_lgy_ddpm(
        #     model_vae, tokenizer_encoder, tokenizer_decoder, args, 1,
        #     ddpm, model_ppl, tokenizer_ppl, z=None, total_step=50
        # )
        # logger.info("PPL = %f", results['ppl'])
        # logger.info("sBLEU = %f", results['sbleu'])
    # if True:
    #     results = calc_rec_lgy(model_vae, tokenizer_encoder, tokenizer_decoder, args, ns=1)
    #     logger.info("Yelp BLEU = %f", results['bleu'])
    #     args.eval_data_file = '/home/guangyi/guangyi/repo/LatentOps/data/datasets/amazon_data/test.merge'
    #     results = calc_rec_lgy(model_vae, tokenizer_encoder, tokenizer_decoder, args, ns=1)
    #     logger.info("Amazon BLEU = %f", results['bleu'])
if __name__ == "__main__":
    main()
