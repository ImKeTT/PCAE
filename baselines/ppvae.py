from tqdm import tqdm
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss


class PPVAE(nn.Module):
    def __init__(self, Vae, config, device):
        super().__init__()
        self.vae = Vae
        self.device = device
        self.z_enc = nn.Sequential(nn.Linear(config.dim_z, config.dim_z//2), nn.LeakyReLU(0.2, True),
                                   nn.Linear(config.dim_z//2, config.dim_z//4), nn.LeakyReLU(0.2, True))
        self.z_dec = nn.Sequential(nn.Linear(config.dim_bottle, config.dim_z//4), 
                                    nn.Linear(config.dim_z//4, config.dim_z//2),
                                   nn.Linear(config.dim_z//2, config.dim_z))
        self.z2mu = nn.Linear(config.dim_z//4, config.dim_bottle)
        self.z2lv = nn.Linear(config.dim_z//4, config.dim_bottle)
        
        self.label_emb = nn.Linear(config.class_num, config.dim_label)
        self.config = config
        self.freeze_encoder()
        self.freeze_decoder()
        
    def freeze_encoder(self):
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
    def freeze_decoder(self):
        for param in self.vae.decoder.parameters():
            param.requires_grad = False
            
    def generate(self, z, top_k=10, top_p=0.5, temperature=1.0):
        num_samples = z.size(0)
        z = self.z_enc(z)
        mu, logvar = self.z2mu(z), self.z2lv(z)
        z_out = self.vae.reparameterize(mu, logvar, nsamples=1).squeeze(1)
        rec_z = self.z_dec(z_out)
        z = self.vae.linear(rec_z)
        
        generated =torch.tensor([[self.vae.tokenizer.bos_token_id]] * num_samples, device=self.device)
        while generated.shape[1] < 1000:
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
        
            
    def forward(self, x, labels, x_attention_mask, decoder_inputs, decoder_inputs_mask=None, beta=1.):
        encoder_outputs = self.vae.encoder(x, x_attention_mask)
        pooled_hidden_fea = self.vae.pooler(encoder_outputs[0])  # model outputs are always tuple in pytorch-transformers (see doc)
        mu_in, logvar_in = self.vae.z_linear(pooled_hidden_fea).chunk(2, -1)
        
        z_in = self.vae.reparameterize(mu_in, logvar_in, nsamples=1)
        z_in = z_in.squeeze(1)
        z = self.z_enc(z_in)
        mu, logvar = self.z2mu(z), self.z2lv(z)
        
        loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        kl_mask = (loss_kl > self.config.dim_target_kl).float()
        loss_kl = (kl_mask * loss_kl).sum(dim=1)
        
        z_out = self.vae.reparameterize(mu, logvar, nsamples=1).squeeze(1)
        rec_z = self.z_dec(z_out)
        loss_z_rec = torch.mean(torch.pow(z_in - rec_z, 2))

        # pdb.set_trace()
        z = self.vae.linear(rec_z)
        decoder_outputs = self.vae.decoder(input_ids=decoder_inputs, attention_mask=decoder_inputs_mask, 
                                       encoder_hidden_states=z)
        logits = self.vae.lm_head(decoder_outputs[0])
        
        loss_rec = None
        loss_fct = CrossEntropyLoss()
        loss_rec = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss = loss_z_rec + loss_rec + beta * loss_kl 
        return loss_z_rec, loss_rec, loss_kl, loss