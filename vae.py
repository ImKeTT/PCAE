#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: vae.py
@author: ImKe
@email: tuisaac163@gmail.com
@feature: #Enter features here
Modified from https://github.com/ChunyuanLI/Optimus/blob/master/code/examples/big_ae/modules/vae.py
"""
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class BartPooler(nn.Module):
    def __init__(self, config):
        super(BartPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class VAE(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder, tokenizer, args, device):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.config = self.decoder.config
        self.pooler = BartPooler(self.config)
        
        padding_idx, vocab_size = tokenizer.pad_token_id, self.config.vocab_size
        self.vocab_size = vocab_size
        self.shared = nn.Embedding(self.vocab_size, self.config.d_model, padding_idx)

        self.args = args
        self.nz = args.dim_z
        self.z_linear = nn.Linear(self.config.hidden_size, 2 * self.nz, bias=False)

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = padding_idx
        self.device = device


        # connector: from Bert hidden units to the latent space
        # self.linear = nn.Linear(args.nz, 2 * args.nz, bias=False)

        # Standard Normal prior
        loc = torch.zeros(self.nz, device=device)
        scale = torch.ones(self.nz, device=device)
        self.prior = torch.distributions.normal.Normal(loc, scale)
        
        if self.args.zmanner=="mem":
            self.linear = nn.Linear(self.nz, self.config.hidden_size * self.config.decoder_layers, bias=False)
        elif self.args.zmanner=="hidden":
            self.linear = nn.Linear(self.nz, self.config.hidden_size, bias=False)
            
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(self.decoder.config.d_model, self.shared.num_embeddings, bias=False)

    def connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)
        mean, logvar = self.z_linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return z, KL

    def connect_deterministic(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.z_linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        logvar.fill_(.0)
        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)
    
    def cond_gen(self, logits, labels):
        masked_lm_loss = None
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss
    
    def rec_loss(self, x, target):
        loss = F.cross_entropy(
            x.transpose(1, 2),
            target,
            ignore_index=self.pad_token_id,
            reduction="none",
        )
        return loss
    
    def build_past(self, z):
        projection = self.linear(z)
        
        cross_attn = projection.reshape(
            self.config.decoder_layers,
            projection.shape[0],
            self.config.decoder_attention_heads,
            1,
            int(self.config.hidden_size / self.config.decoder_attention_heads)
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values
    
    def generate(self, z, top_k=10, top_p=0.5, use_cache=None, temperature=1.0):
        num_samples = z.size(0)
        if self.args.zmanner == "mem":
            z = self.build_past(z)
            generated = torch.tensor([[self.tokenizer.bos_token_id]] * num_samples, device=device)
            while generated.shape[1] < 1000:
                decoder_outputs = self.decoder(input_ids=generated, past_key_values=z, use_cache=use_cache)
                z = decoder_outputs[1] if use_cache is not None else z
                logits = self.lm_head(decoder_outputs[0])
                logits = logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering_batch(logits, top_k=top_k, top_p=top_p)

                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, 1)
                # next_token_id = torch.argmax(filtered_logits, dim=-1).unsqueeze(0)

                generated = torch.cat((generated, next_token_id), dim=1)
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        elif self.args.zmanner == "hidden":
            z = self.linear(z)
            generated =torch.tensor([[self.tokenizer.bos_token_id]] * num_samples, device=device)
            while generated.shape[1] < 1000:
                decoder_outputs = self.decoder(input_ids=generated, encoder_hidden_states=z, use_cache=use_cache)
                logits = self.lm_head(decoder_outputs[0])
                logits = logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering_batch(logits, top_k=top_k, top_p=top_p)

                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, 1)
                # next_token_id = torch.argmax(filtered_logits, dim=-1).unsqueeze(0)

                generated = torch.cat((generated, next_token_id), dim=1)
                not_finished = next_token_id != self.tokenizer.eos_token_id
                if torch.sum(not_finished) == 0:
                    break
#                 if next_token_id == self.tokenizer.eos_token_id:
#                     break
        return generated
        

    def forward(self, inputs, labels, attention_mask, decoder_inputs, decoder_inputs_mask=None, beta=1.):
        reconstrution_mask=(labels != self.pad_token_id).float()
        sent_length = torch.sum(reconstrution_mask, dim=1)

        
        encoder_outputs = self.encoder(inputs, attention_mask)
        pooled_hidden_fea = self.pooler(encoder_outputs[0])  # model outputs are always tuple in pytorch-transformers (see doc)

        if self.args.fb_mode==0: 
            # Connect hidden feature to the latent space
            latent_z, loss_kl = self.connect(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)
            if self.args.zmanner == "mem":
                past = self.build_past(latent_z)
                past_length = 1 # past[0][0].size(-2)
                # Decoding
                decoder_inputs = decoder_inputs[:, -1:]
                decoder_outputs = self.decoder(input_ids=decoder_inputs, attention_mask=decoder_inputs_mask, 
                                       encoder_hidden_states=encoder_outputs[0], past_key_values=past)
            elif self.args.zmanner == "hidden":
                z = self.linear(latent_z)
                decoder_outputs = self.decoder(input_ids=decoder_inputs, attention_mask=decoder_inputs_mask, 
                                       encoder_hidden_states=z)
            logits = self.lm_head(decoder_outputs[0])
            if self.args.zmanner == "mem":
                logits = logits.repeat(1, labels.size(-1), 1)
            loss_rec =  self.cond_gen(logits, labels) # model outputs are always tuple in pytorch-transformers (see doc)
    
        elif self.args.fb_mode==1:  
            # Connect hidden feature to the latent space
            mu, logvar = self.z_linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = self.reparameterize(mu, logvar, nsamples=1)
            latent_z = latent_z.squeeze(1)
            loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
            kl_mask = (loss_kl > self.args.dim_target_kl).float()
            loss_kl = (kl_mask * loss_kl).sum(dim=1)

            # pdb.set_trace()
            if self.args.zmanner == "mem":
                past = self.build_past(latent_z)
                past_length = 1 # past[0][0].size(-2)
                # Decoding
                decoder_inputs = decoder_inputs[:, -1:]
                decoder_outputs = self.decoder(input_ids=decoder_inputs, attention_mask=decoder_inputs_mask, 
                                       encoder_hidden_states=encoder_outputs[0], past_key_values=past)
            elif self.args.zmanner == "hidden":
                z = self.linear(latent_z)
                decoder_outputs = self.decoder(input_ids=decoder_inputs, attention_mask=decoder_inputs_mask, 
                                       encoder_hidden_states=z)
            logits = self.lm_head(decoder_outputs[0])
            if self.args.zmanner == "mem":
                logits = logits.repeat(1, labels.size(-1), 1)
            loss_rec =  self.cond_gen(logits, labels) # model outputs are always tuple in pytorch-transformers (see doc)

            
        # pdb.set_trace()
        if self.args.length_weighted_loss:
            loss = loss_rec / sent_length + beta * loss_kl
        else:
            loss = loss_rec + beta * loss_kl 


        return loss_rec, loss_kl, loss