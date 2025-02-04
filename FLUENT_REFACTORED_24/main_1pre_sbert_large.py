from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from torch import nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data import qa_paired, qa_paired_eval
from architecture_1pre_largeenc import FLUENTSOTA
from evaluation_tool import calculate_bleu, count_bleu_score, compute_average_chrf, generate_predictions
from neptune_fluent import Neptune_Fluent 

import torch
import matplotlib.pyplot as plt
import time
import pandas as pd

encoder_id = 'denaya/indoSBERT-large'
print("initiliazing encoder model and tokenizer : {}".format(encoder_id))
enc_tokenizer = AutoTokenizer.from_pretrained(encoder_id, clean_up_tokenization_spaces=True)
enc_model = AutoModel.from_pretrained(encoder_id)

decoder_id = 'indonesian-nlp/gpt2-medium-indonesian'
# decoder_id = 'cahya/gpt2-large-indonesian-522M'
print("initiliazing decoder model and tokenizer : {}".format(decoder_id))
dec_model = GPT2LMHeadModel.from_pretrained(decoder_id)
dec_tokenizer = GPT2Tokenizer.from_pretrained(decoder_id, clean_up_tokenization_spaces=True)

dec_tokenizer.add_tokens(['[PRE1]'])
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

dec_size = 1280 if 'large' in decoder_id else 1024
model = FLUENTSOTA(enc_model, dec_model, enc_tokenizer, dec_tokenizer, dec_size)
model.to(device)

all_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
untrainable_params = all_params - trainable_params
print(f'All parameters: {all_params:,}')
print(f'Trainable parameters: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)')
print(f'Untrainable parameters: {untrainable_params:,} ({untrainable_params/all_params*100:.2f}%)')

bleu_score_eval = pd.DataFrame(columns=['Epoch', '1-gram', '2-gram', '3-gram', '4-gram', 'cumulative-1-gram', 'cumulative-2-gram', 'cumulative-3-gram', 'cumulative-4-gram'])
bleu_score_train = pd.DataFrame(columns=['Epoch', '1-gram', '2-gram', '3-gram', '4-gram', 'cumulative-1-gram', 'cumulative-2-gram', 'cumulative-3-gram', 'cumulative-4-gram'])

questions = qa_paired['Pertanyaan'].apply(lambda x: x.lower().replace('[BOS]', '').replace('[EOS]', '')).to_list()
answers = qa_paired['Jawaban'].apply(lambda x: x.replace('[BOS]', '').replace('[EOS]', '').lower().strip()).to_list()
questions_eval = qa_paired_eval['Pertanyaan'].apply(lambda x: x.lower().replace('[BOS]', '').replace('[EOS]', '')).to_list()
answers_eval = qa_paired_eval['Jawaban'].apply(lambda x: x.replace('[BOS]', '').replace('[EOS]', '').lower().strip()).to_list()

epochs = 500
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
run = Neptune_Fluent.mulai(encoder_id, decoder_id, num_pre_token=0)
bleu_result_eval = {"cumulative-4-gram":0}
bleu_result_train = {"cumulative-4-gram":0}
chrf_result_eval = 0
chrf_result_train = 0

print("start training")
for ep in range(epochs):
    torch.cuda.empty_cache()

    total_loss = 0
    for instance in qa_paired.iterrows():
        # print("---------------")
        optimizer.zero_grad()

        pertanyaan = instance[1]['Pertanyaan']
        jawaban = instance[1]['Jawaban']
        jawaban_withpre = '[PRE1]' + jawaban

        tokenized_jawaban_withpre = model.dec_tokenizer(jawaban_withpre)
        tokenized_jawaban_withpre = torch.tensor(tokenized_jawaban_withpre['input_ids']).unsqueeze(0)

        enc_logits = model.encoding(pertanyaan)
        output = model.decoding_train(enc_logits, target=jawaban, target_with_pre=tokenized_jawaban_withpre)

        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    run["train/loss"].append(total_loss)
    print(f'Epoch {ep+1}/{epochs} - Loss: {total_loss:.4f}')
    if (ep+1) % 10 == 0:
        print(f'\n-----------------------------------------')
        test_question = qa_paired['Pertanyaan'].iloc[0]
        outputs = model.generate(test_question)
        decoded_output = model.dec_tokenizer.decode(outputs[0])
        print(f'Q >>> {test_question}')
        print(f'A <<< {decoded_output}')
        test_question = qa_paired['Pertanyaan'].iloc[1]
        outputs = model.generate(test_question)
        decoded_output = model.dec_tokenizer.decode(outputs[0])
        print(f'Q >>> {test_question}')
        print(f'A <<< {decoded_output}')
        test_question = qa_paired['Pertanyaan'].iloc[4]
        outputs = model.generate(test_question)
        decoded_output = model.dec_tokenizer.decode(outputs[0])
        print(f'Q >>> {test_question}')
        print(f'A <<< {decoded_output}')
        print(f'-----------------------------------------\n')

    if (ep+1) % 10 == 0:
        preds_eval = generate_predictions(model, questions_eval)

        bleu_result_eval = calculate_bleu(preds_eval, questions_eval, answers_eval)
        bleu_score_eval = pd.concat([bleu_score_eval, pd.DataFrame({'Epoch': ep+1, **bleu_result_eval}, index=[len(bleu_score_eval)])], ignore_index=True)
        print(f'BLEU Score Eval: {bleu_result_eval["cumulative-4-gram"]:.4f}\n')

        chrf_result_eval = compute_average_chrf(preds_eval, answers_eval)
        print(f'CHRF Score Eval: {chrf_result_eval:.4f}\n')

        preds_train = generate_predictions(model, questions)
        bleu_result_train = calculate_bleu(preds_train, questions, answers)
        bleu_score_train = pd.concat([bleu_score_train, pd.DataFrame({'Epoch': ep+1, **bleu_result_train}, index=[len(bleu_score_train)])], ignore_index=True)
        print(f'BLEU Score Train: {bleu_result_train["cumulative-4-gram"]:.4f}\n')

        chrf_result_train = compute_average_chrf(preds_train, answers)
        print(f'CHRF Score Train: {chrf_result_train:.4f}\n')

    run["eval/chrf"].append(chrf_result_eval)
    run["train/chrf"].append(chrf_result_train)
    run["eval/bleu"].append(bleu_result_eval["cumulative-4-gram"])
    run["train/bleu"].append(bleu_result_train["cumulative-4-gram"])

run.stop()
print("finished training")
