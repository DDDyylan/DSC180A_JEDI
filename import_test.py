from __future__ import absolute_import, division, print_function
import sys
sys.path.append("./src")
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
import sys

from train_ddpm_latent import calc_ppl_lgy_ddpm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pudb.remote import set_trace
import shutil, sys

from modules import sample_sequence_conditional


from transformers import LlamaTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType