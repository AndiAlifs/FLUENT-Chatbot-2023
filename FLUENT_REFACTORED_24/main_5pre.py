from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from torch import nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data import qa_paired, qa_paired_eval
from architecture_5pre import FLUENTSOTA
import torch
import matplotlib.pyplot as plt
import time

encoder_id = 'firqaaa/indo-sentence-bert-base'
print("initiliazing encoder model and tokenizer : {}".format(encoder_id))
enc_tokenizer = AutoTokenizer.from_pretrained(encoder_id, clean_up_tokenization_spaces=True)
enc_model = AutoModel.from_pretrained(encoder_id)

decoder_id = 'indonesian-nlp/gpt2-medium-indonesian'
print("initiliazing decoder model and tokenizer : {}".format(decoder_id))
dec_model = GPT2LMHeadModel.from_pretrained(decoder_id)
dec_tokenizer = GPT2Tokenizer.from_pretrained(decoder_id, clean_up_tokenization_spaces=True)

dec_tokenizer.add_tokens(['[PRE1]'])
dec_tokenizer.add_tokens(['[PRE2]'])
dec_tokenizer.add_tokens(['[PRE3]'])
dec_tokenizer.add_tokens(['[PRE4]'])
dec_tokenizer.add_tokens(['[PRE5]'])
dec_tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                    'bos_token': '[BOS]',
                                    'eos_token': '[EOS]',
                                    'sep_token': '[SEP]',})
dec_model.config.pad_token_id = dec_tokenizer.pad_token_id
dec_model.config.bos_token_id = dec_tokenizer.bos_token_id
dec_model.config.eos_token_id = dec_tokenizer.eos_token_id
dec_model.config.sep_token_id = dec_tokenizer.sep_token_id
dec_model.resize_token_embeddings(len(dec_tokenizer))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc_model = enc_model.to(device)
dec_model = dec_model.to(device)
print("finished initiliazing encoder model and tokenizer")

for param in enc_model.parameters():
    param.requires_grad = False

for param in dec_model.parameters():
    param.requires_grad = False

for param in dec_model.transformer.h[:-15].parameters():
    param.requires_grad = True

for param in enc_model.encoder.layer[:-15].parameters():
    param.requires_grad = True

print("Encoder Trainable Parameters : {}%".format(sum(p.numel() for p in enc_model.parameters() if p.requires_grad)/sum(p.numel() for p in enc_model.parameters())*100))
print("Decoder Trainable Parameters : {}%".format(sum(p.numel() for p in dec_model.parameters() if p.requires_grad)/sum(p.numel() for p in dec_model.parameters())*100))


