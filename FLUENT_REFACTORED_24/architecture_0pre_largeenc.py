import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FLUENTSOTA(nn.Module):
    def __init__(self, enc_model, dec_model, enc_tokenizer, dec_tokenizer, max_length=200, dec_size=1024):
        super(FLUENTSOTA, self).__init__()
        self.enc_model = enc_model
        self.dec_model = dec_model
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.enc_mapper = nn.Linear(1024, dec_size)
        self.enc_mapper2 = nn.Linear(dec_size, dec_size)

        self.prefix_param = nn.Parameter(torch.randn(1, 1, dec_size))  # Learnable parameter for prefix 

        self.prefix_nn = nn.Linear(dec_size, dec_size)
        self.max_length = max_length
    
    def encoding(self, sentence):
        tokens = self.enc_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        tokens = tokens.to(device)
        with torch.no_grad():
            output = self.enc_model(**tokens)
        enc_logits = output.last_hidden_state.sum(dim=1)
        enc_logits = self.enc_mapper(enc_logits).to(device)
        enc_logits = self.enc_mapper2(enc_logits).to(device)
        return enc_logits
    
    def get_prefix(self, batch_size=1):
        # Process the learnable parameter through prefix_nn layers
        prefix = self.prefix_param.expand(batch_size, 1, -1)
        prefix = self.prefix_nn(prefix)
        return prefix

    def get_embedding(self, sentence):
        tokens = self.dec_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        tokens = tokens.to(device)
        wte = self.dec_model.get_input_embeddings()
        return wte(tokens['input_ids'])

    def dec_tokenizer(self, sentence):
        tokens = self.dec_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        return tokens

    def decoding_train(self, enc_logits, target, target_with_pre):
        prefix = self.get_prefix(batch_size=enc_logits.size(0))
        embed = self.get_embedding(target)
        pref_with_embed = torch.cat((prefix, embed), dim=1)
        output = self.dec_model(inputs_embeds=pref_with_embed, labels=target_with_pre)
        return output

    def generate(self, quest):
        enc_logits = self.encoding(quest)
        prefix_se = '[BOS]'
        prefix_dec_embed = self.get_embedding(prefix_se)
        # prefixs = self.add_prefix(enc_logits)
        
        # pref_with_embed = torch.cat((enc_logits.unsqueeze(dim=0), prefix_dec_embed), dim=1)

        output = self.dec_model.generate(   inputs_embeds=prefix_dec_embed, 
                                            max_length=self.max_length, 
                                            pad_token_id=self.dec_model.config.eos_token_id)
        returned_output = []
        for i in range(len(output[0])):
            if output[0][i] != self.dec_model.config.eos_token_id:
                returned_output.append(output[0][i])
            else:
                break
        return torch.tensor(returned_output).unsqueeze(0)