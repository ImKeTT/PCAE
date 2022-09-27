#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: optimus.py
@author: ImKe
@email: tuisaac163@gmail.com
@feature: #Enter features here
Optimus Baseline from https://github.com/ChunyuanLI/Optimus/blob/master/code/examples/big_ae/run_lm_vae_label_ctrl_gen.py
"""
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import sys
sys.path.append("../")
from utils import *


class Optimus(nn.Module):
    def __init__(self, Vae, config, device, layer_num=5):
        super().__init__()
        self.vae = Vae
        self.device = device
        self.layer_num = layer_num
        self.latent_generator = nn.Linear(config.dim_z, config.dim_z)
        self.latent_discriminator = nn.Linear(config.dim_z, 1)
        self.label_emb = nn.Linear(config.class_num, config.dim_label)
        self.label_linear = nn.Linear(config.dim_label, config.dim_z)
        self.config = config
        
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        
    def freeze_encoder(self):
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
    def freeze_decoder(self):
        for param in self.vae.decoder.parameters():
            param.requires_grad = False
            
    def generate(self, z, y, top_k=10, top_p=0.5, temperature=1.0):
        num_samples = z.size(0)
        y = torch.tensor(y, device=self.device)
        label = F.one_hot(y.squeeze(0).long(), self.config.class_num).float().to(self.device)
        label = self.label_emb(label)
        label = self.label_linear(label)
        
        z_label = z + label
        # pdb.set_trace()
        z = self.vae.linear(z_label)
        
        generated =torch.tensor([[self.vae.tokenizer.bos_token_id]] * num_samples, device=self.device)
        while generated.shape[1] < 80:
            decoder_outputs = self.vae.decoder(input_ids=generated, encoder_hidden_states=z)
            logits = self.vae.lm_head(decoder_outputs[0])
            logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering_batch(logits, top_k=top_k, top_p=top_p)

            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, 1)
            # next_token_id = torch.argmax(filtered_logits, dim=-1).unsqueeze(0)

            generated = torch.cat((generated, next_token_id), dim=1)
            not_finished = next_token_id != self.vae.tokenizer.eos_token_id
            if torch.sum(not_finished) == 0:
                break
        return generated
        
            
    def forward(self, x, y, lm_labels, x_attention_mask, decoder_inputs, decoder_inputs_mask=None, alpha=1., beta=1.):
        ones_label = torch.ones_like(y).to(dtype=torch.float32)
        zeros_label = torch.zeros_like(y).to(dtype=torch.float32)
        random_noise = torch.nn.init.normal_(torch.empty(x.size(0), self.config.dim_z)).to(device=self.device, dtype=torch.float32)
        
        encoder_outputs = self.vae.encoder(x, x_attention_mask)
        pooled_hidden_fea = self.vae.pooler(encoder_outputs[0])  # model outputs are always tuple in pytorch-transformers (see doc)
        mu, logvar = self.vae.z_linear(pooled_hidden_fea).chunk(2, -1)
        
        label = F.one_hot(y.squeeze(0).long(), self.config.class_num).float().to(self.device)
        label = self.label_emb(label)
        label = self.label_linear(label)
        
        latent_z = self.vae.reparameterize(mu, logvar, nsamples=1)
        z = latent_z.squeeze(1)
        loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        kl_mask = (loss_kl > self.config.dim_target_kl).float()
        loss_kl = (kl_mask * loss_kl).sum(dim=1)
        
        ## fake z
        gen_z = self.latent_generator(random_noise)
        # gen_z = random_noise

        #################### Latent discriminator for sampling from a simple distribution ####################
        prob_encode_z_dis = self.latent_discriminator(z).squeeze(1).float()  # (B)
        prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()  # (B)
        # Train latent discriminator
        loss_lsd = self.BCEWithLogitsLoss(prob_gen_z_dis, zeros_label) + self.BCEWithLogitsLoss(prob_encode_z_dis, ones_label)
        acc_encode_z_dis = ((prob_encode_z_dis >= 0).float() == ones_label).float()
        acc_gen_z_dis = ((prob_gen_z_dis >= 0).float() == zeros_label).float()
        # Train sampler adversarially
        loss_lsg = self.BCEWithLogitsLoss(prob_gen_z_dis, ones_label)
        
        z_label = z + label

        # pdb.set_trace()
        z = self.vae.linear(z_label)
        decoder_outputs = self.vae.decoder(input_ids=decoder_inputs, attention_mask=decoder_inputs_mask, 
                                       encoder_hidden_states=z)
        logits = self.vae.lm_head(decoder_outputs[0])
        
        loss_rec = None
        loss_rec = self.CrossEntropyLoss(logits.view(-1, logits.size(-1)), lm_labels.view(-1))
        
        loss = loss_rec + beta * loss_kl + alpha * loss_lsg
        return loss_rec, loss_kl, loss_lsd, loss