from vae import VAE
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss


class BroadcastingNet(nn.Module):
    def __init__(self, emb_size, z_size, class_num, layer_num=5):
        """
        Broadcasting Net in the original paper
        """
        super().__init__()
#         self.es = emb_size
#         self.hs = hid_size
#         self.z_size = z_size
        self.lm = layer_num
        self.forwardnet = nn.ModuleList()
        self.act = nn.ReLU()
        for i in range(layer_num):
            self.forwardnet.append(nn.Linear(z_size+emb_size, z_size))
    
    def forward(self, label, z):
        label_emb = label
        for i in range(self.lm):
            z = torch.cat([label_emb, z], -1)
            z = self.forwardnet[i](z)
#             z = self.act(z)
        return z

class PCAE(nn.Module):
    def __init__(self, Vae, config, device, layer_num=5):
        super().__init__()
        self.vae = Vae
        self.device = device
        self.layer_num = layer_num
        self.lper = BroadcastingNet(config.dim_label, config.dim_z, config.class_num, self.layer_num)
        self.label_emb = nn.Linear(config.class_num, config.dim_label)
        self.config = config
        self.freeze_encoder()
#         self.freeze_decoder()
        
    def freeze_encoder(self):
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
    def freeze_decoder(self):
        for param in self.vae.decoder.parameters():
            param.requires_grad = False
            
    def generate(self, z, y, top_k=50, top_p=0.5, temperature=1.0):
        num_samples = z.size(0)
        labels = torch.full([num_samples, 1], y).to(self.device)
        labels = torch.stack([torch.eye(self.config.class_num)[label.squeeze(0)].to(self.device) for label in labels])
        labels = self.label_emb(labels)
        
        z = self.lper(labels, z)
        z = self.vae.linear(z)
        
        generated =torch.tensor([[self.vae.tokenizer.bos_token_id]] * num_samples, device=self.device)
        while generated.shape[1] <= 50:
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
        
            
    def forward(self, x, y, labels, x_attention_mask, decoder_inputs, use_mean, decoder_inputs_mask=None, beta=1.):
        encoder_outputs = self.vae.encoder(x, x_attention_mask)
        pooled_hidden_fea = self.vae.pooler(encoder_outputs[0])  # model outputs are always tuple in pytorch-transformers (see doc)
        mu, logvar = self.vae.z_linear(pooled_hidden_fea).chunk(2, -1)
        
        label = F.one_hot(y.squeeze(0).long(), self.config.class_num).float().to(self.device)
        label = self.label_emb(label)
        mu_label = self.lper(label, mu)
        
        latent_z = self.vae.reparameterize(mu_label, logvar, nsamples=1)
        z_label = latent_z.squeeze(1)
        loss_kl = 0.5 * (mu_label.pow(2) + logvar.exp() - logvar - 1)
        kl_mask = (loss_kl > self.config.dim_target_kl).float()
        loss_kl = (kl_mask * loss_kl).sum(dim=1)

        # pdb.set_trace()
        if not use_mean:
            z = self.vae.linear(z_label)
        else:
            z = mu_label
            z = self.vae.linear(z)
        decoder_outputs = self.vae.decoder(input_ids=decoder_inputs, attention_mask=decoder_inputs_mask, 
                                       encoder_hidden_states=z)
        logits = self.vae.lm_head(decoder_outputs[0])
        
        loss_rec = None
        loss_fct = CrossEntropyLoss()
        loss_rec = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss = loss_rec + beta * loss_kl 
        return loss_rec, loss_kl, loss