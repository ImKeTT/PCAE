import torch
from vae import VAE
from pcae import PCAE
from baselines.ppvae import PPVAE
from baselines.optimus import Optimus
from tqdm import tqdm
from transformers import BartTokenizer, BartModel, AdamW, BartForConditionalGeneration
from utils import *
import torch.nn as nn
import torch
from logger import Logger
import datetime, math, os, sys, json, argparse, time, re, copy
import numpy as np

parser = argparse.ArgumentParser()
## data preparation
parser.add_argument("--dataset", default='', type=str, required=False, choices=["yelp", "yahoo", "titles"],
                    help="Training dataset.")
parser.add_argument('--no_gpu', action='store_true')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])
parser.add_argument('--seed', default=42, type=int)
parser.add_argument("--per_gpu_train_batch_size", default=42, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=5, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--dim_label", default=8, type=int, help="Dim of label embedding layer")
parser.add_argument("--zmanner", default='hidden', type=str, choices=['hidden', 'mem'])
parser.add_argument("--dim_z", default=128, type=int, help="Dim of latent space")

parser.add_argument("--bart_version", default='facebook/bart-base', type=str)
parser.add_argument('--use_mean', action='store_true', help="Use mean representation of latent space in PCAE")
parser.add_argument('--length_weighted_loss', action='store_true')
parser.add_argument("--gen_batch_size", default=5, type=int,
                    help="Batch size per GPU/CPU for generation.")
parser.add_argument("--workers", default=3, type=int,
                    help="Dataloader worker.")
parser.add_argument("--lr", default=3e-5, type=float, help="The initial learning rate.")
parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--beta", default=1., type=float)
parser.add_argument("--train_epochs", default=10, type=int, help="Training Epoch for Finetuning.")
parser.add_argument("--plugin_train_epochs", default=20, type=int, help="Training Epoch for Plugin Training.")

parser.add_argument("--fb_mode", default=1, type=int, help="Free bit threshold mode.")
parser.add_argument("--layer_num", default=10, type=int, help="Broadcasting Layer Number of PCAE.")
parser.add_argument("--dim_target_kl", default=0.1, type=float, help="KL thresh for each dimension in VAE.")
parser.add_argument("--gen_k", default=100, type=int, help="Number of batch sentence to generate.")
parser.add_argument("--task", default='sentiment', type=str, choices=['sentiment', 'tense', 'topics', 'quess_s', 'quess'])
parser.add_argument("--task_label", default='pos', type=str, help="For PPVAE only")
parser.add_argument("--ppvae_dim_bottle", default=25, type=int, help="For PPVAE only")
parser.add_argument("--ppvae_loss_relax", default=10, type=float, help="For PPVAE only")
parser.add_argument("--sample_n", default=100, type=int, help="Number of training instance for each class for training.")

parser.add_argument("--run_mode", default='pcae', type=str, choices=['vae_ft', 'ppvae', 'pcae', 'optimus'])

parser.add_argument('--first_token_pooling', action='store_true', 
    help='Use the first token as the pooling signal in VAE, else the mean pooling.')



## Data setup details for PCAE plugin training
task_dataset_dict = {"sentiment": "yelp", "tense": "yelp", "topics": "titles", "quess_s": "yahoo", "quess": "yahoo"}
class_num_dataset_dict = {"sentiment": 2, "tense": 2, "topics": 4, "quess_s": 6, "quess": 10}

def evaluate_vae(dataloader, model, tokenizer, device, logger):
    model.eval()
    losses = []
    losses_rec = []
    losses_kl = []
    for batch_id, texts in enumerate(tqdm(dataloader)):
        out = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)
        pad_token_id = tokenizer.pad_token_id
        y = out['input_ids']
        y_ids = y[:, :-1].contiguous()
        y_mask = out['attention_mask'][:, :-1]
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100

        loss_rec, loss_kl, loss = model(out['input_ids'].to(device), labels=lm_labels.to(device), 
                                                   decoder_inputs=y_ids.to(device), 
                                                   attention_mask=out['attention_mask'].to(device))
        loss_rec, loss_kl, loss = loss_rec.mean(), loss_kl.mean(), loss.mean()
        losses.append(loss.detach().cpu().numpy())
        losses_rec.append(loss_rec.detach().cpu().numpy())
        losses_kl.append(loss_kl.detach().cpu().numpy())

    logger.info("Val Loss     : {:.4f}".format(np.mean(losses)))
    logger.info("Val Loss Rec : {:.4f}".format(np.mean(losses_rec)))
    logger.info("Val Loss KL. : {:.4f}".format(np.mean(losses_kl)))
    return np.mean(losses_rec)

def evaluate_pcae(dataloader, model, tokenizer, device, logger):
    model.eval()
    losses = []
    losses_rec = []
    losses_kl = []
    for batch_id, (x, ylabel) in enumerate(tqdm(dataloader)):
        out = tokenizer.batch_encode_plus(x, return_tensors="pt", padding=True)
        pad_token_id = tokenizer.pad_token_id
        y = out['input_ids']
        y_ids = y[:, :-1].contiguous()
        y_mask = out['attention_mask'][:, :-1]
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100

        loss_rec, loss_kl, loss = model(out['input_ids'].to(device), ylabel, lm_labels.to(device), out['attention_mask'].to(device), 
                                        y_ids.to(device))
        loss_rec, loss_kl, loss = loss_rec.mean(), loss_kl.mean(), loss.mean()
        losses.append(loss.detach().cpu().numpy())
        losses_rec.append(loss_rec.detach().cpu().numpy())
        losses_kl.append(loss_kl.detach().cpu().numpy())

    logger.info("Val Loss     : {:.4f}".format(np.mean(losses)))
    logger.info("Val Loss Rec : {:.4f}".format(np.mean(losses_rec)))
    logger.info("Val Loss KL. : {:.4f}".format(np.mean(losses_kl)))
    return np.mean(losses_rec)

def VAE_finetuning(args):
    gpu = not args.no_gpu
    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    ## 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(args.bart_version)
    model = BartForConditionalGeneration.from_pretrained(args.bart_version)
    model.to(device)

    # epos = [8, 10, 10]
    # bss = [64, 64, 64]
    dataname = args.dataset
    vae = VAE(model.model.encoder, model.model.decoder, tokenizer, args, device)
    vae.shared = model.model.shared
    vae.lm_head = model.lm_head
    if len(args.gpu)>1:
        vae = nn.DataParallel(vae, device_ids=args.gpu)
    vae.to(device)
    os.makedirs(f"checkpoints/{dataname}/BART", exist_ok=True)
    log_file = f"checkpoints/{dataname}/BART/log_e{args.train_epochs}_vae_z{args.dim_z}_{args.zmanner}.txt"
    logger = Logger(log_file)
    train_data = read_txt(f"data/{dataname}/train.txt")
    val_data = read_txt(f"data/{dataname}/valid.txt")
    test_data = read_txt(f"data/{dataname}/valid.txt")

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, shuffle=True)
    iterations = args.train_epochs * len(train_loader)
    print(f"Iterations: {iterations}")
    betas = frange_cycle_zero_linear(iterations, start=0.0, stop=1.0,  n_cycle=4, 
                                     ratio_increase=0.2, ratio_zero=0.1)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, shuffle=True)
    optimizer = AdamW(vae.parameters(), lr=args.lr, correct_bias=True)

    ## Fine-tuning
    best_val_loss = 99999.
    total_iters = 0
    for e in range(args.train_epochs):
        vae.train()
        losses = []
        losses_rec = []
        losses_kl = []
        for batch_id, texts in enumerate(tqdm(train_loader)):
            beta = betas[total_iters]
            out = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True)
            pad_token_id = tokenizer.pad_token_id
            y = out['input_ids']
            y_ids = y[:, :-1].contiguous()
            y_mask = out['attention_mask'][:, :-1]
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            
            loss_rec, loss_kl, loss = vae(out['input_ids'].to(device), 
                                            labels=lm_labels.to(device), decoder_inputs=y_ids.to(device), 
                                            attention_mask=out['attention_mask'].to(device), beta=beta)
            
            loss_rec, loss_kl, loss = loss_rec.mean(), loss_kl.mean(), loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().numpy())
            losses_rec.append(loss_rec.detach().cpu().numpy())
            losses_kl.append(loss_kl.detach().cpu().numpy())
            total_iters += 1
            
        logger.info("Train Loss     : {:.4f}".format(np.mean(losses)))
        logger.info("Train Loss Rec : {:.4f}".format(np.mean(losses_rec)))
        logger.info("Train Loss KL. : {:.4f}".format(np.mean(losses_kl)))
        val_loss = evaluate_vae(val_loader, vae, tokenizer, device, logger)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info("Saving the Best Eval Weights..")
            save_orderdict = vae.state_dict()
            torch.save(save_orderdict, f"checkpoints/{dataname}/BART/best_val_vae_{args.zmanner}.pt")

def Optimus_plugin_fintuning(args):
    gpu = not args.no_gpu
    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    dataset = task_dataset_dict[args.task]

    ## setup config for each training task
    class config():
        pass
    config.fb_mode=args.fb_mode ## 1
    config.dim_target_kl=args.dim_target_kl ## 0.1
    config.zmanner=args.zmanner ## "hidden"
    config.dim_label = args.dim_label ## 8
    config.dim_z=args.dim_z ## 128
    config.class_num = class_num_dataset_dict[args.task]
    config.train = f"data/{dataset}/{args.task}/label_text{args.sample_n}.txt"
    config.test = f"data/{dataset}/{args.task}/label_text100.txt"
    config.lr = args.lr ## 1e-4
    config.epoch=args.plugin_train_epochs ## 28
    config.gen_k = args.gen_k ## 100

    tokenizer = BartTokenizer.from_pretrained(args.bart_version)
    model = BartForConditionalGeneration.from_pretrained(args.bart_version)

    vae = VAE(model.model.encoder, model.model.decoder, tokenizer, config, device)
    state = torch.load(f"checkpoints/{dataset}/BART/best_val_vae_hidden.pt")
    if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
        state_copy = copy.copy(state)
        keys = state_copy.keys()
        for k in keys:
            state[k.replace('module.', '')] = state.pop(k)
    vae.load_state_dict(state)
    del state
    print("Finish Loading Pre-trained Weights..")
    model = Optimus(vae, config, device)
    model.to(device)

    traindata = ConditionalGenerationDataset.from_file(file_path=config.train)
    trainloader = DataLoader(traindata, batch_size=args.train_batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config.lr, correct_bias=True)
    os.makedirs(f"bart_result/optimus/results/{dataset}", exist_ok=True)
    log_file = f"bart_result/optimus/results/{dataset}/{args.task}-epoch{config.epoch}-bs{args.train_batch_size}-lr{config.lr}-ns{args.sample_n}.log"
    logger = Logger(log_file)

    ## Model training
    model.train()
    best_val_loss = 99999.
    total_iters = 0
    for e in range(config.epoch):
        losses = []
        losses_rec = []
        losses_kl = []
        losses_lsd = []
        for batch_id, (x, ylabel) in enumerate(tqdm(trainloader)):
            out = tokenizer.batch_encode_plus(x, return_tensors="pt", padding=True)
            pad_token_id = tokenizer.pad_token_id
            y = out['input_ids']
            y_ids = y[:, :-1].contiguous()
            y_mask = out['attention_mask'][:, :-1]
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            
            ylabel = torch.tensor(ylabel, device=device)
            loss_rec, loss_kl, loss_lsd, loss = model(out['input_ids'].to(device), ylabel, 
                                            lm_labels.to(device), 
                                            out['attention_mask'].to(device), 
                                            y_ids.to(device),  alpha=args.alpha, beta=args.beta)
            
            loss_rec, loss_kl, loss_lsd, loss = loss_rec.mean(), loss_kl.mean(),  loss_lsd.mean(), loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().numpy())
            losses_rec.append(loss_rec.detach().cpu().numpy())
            losses_kl.append(loss_kl.detach().cpu().numpy())
            losses_lsd.append(loss_lsd.detach().cpu().numpy())
            
        logger.info("Train Loss     : {:.4f}".format(np.mean(losses)))
        logger.info("Train Loss Rec : {:.4f}".format(np.mean(losses_rec)))
        logger.info("Train Loss KL. : {:.4f}".format(np.mean(losses_kl)))
        logger.info("Train Loss LSD : {:.4f}".format(np.mean(losses_lsd)))
        total_iters += 1

    ## Generation
    os.makedirs(f"bart_result/optimus/sentences/{dataset}/{args.task}", exist_ok=True)
    for y in range(config.class_num):
        model.eval()
        finalsents = []
        for _ in tqdm(range(config.gen_k)):
            z = torch.randn(args.gen_batch_size, config.dim_z).to(device)
            with torch.no_grad():
                sents = model.generate(z, y)
            texts = []
            for ii in sents:
                endindex = (ii == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(endindex) !=0:
                    texts.append(tokenizer.decode(ii[1: min(endindex)]))
                else:
                    continue
            finalsents.extend(texts)
        with open(f"bart_result/optimus/sentences/{dataset}/{args.task}/{y}-epoch{config.epoch}-bs{args.train_batch_size}-lr{config.lr}-ns{args.sample_n}-{config.gen_k}K.txt", "w") as f:
            for sent in finalsents:
                f.write(sent + "\n")
        f.close()

def PPVAE_plugin_training(args):
    gpu = not args.no_gpu
    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # datali = [100, 300, 500, 800]
    dataset = task_dataset_dict[args.task]
    class config():
        pass
    config.fb_mode=args.fb_mode ##1
    config.dim_target_kl=args.dim_target_kl ##0.1
    config.beta=args.beta ##1.0
    config.zmanner=args.zmanner ##"hidden"
    config.dim_label = args.dim_label ##8
    config.dim_z = args.dim_z ##128
    config.class_num = class_num_dataset_dict[args.task]
    config.alpha = args.alpha ## 0.1
    config.train = f"data/{dataset}/{args.task}/{args.sample_n}.{args.task_label}"

    if args.task == "sentiment":
        if args.task_label == "pos":
            config.neg_train = f"data/{dataset}/{args.task}/{args.sample_n}.neg"
        elif args.task_label == "neg":
            config.neg_train = f"data/{dataset}/{args.task}/{args.sample_n}.pos"
        else:
            raise NotImplementedError
    elif args.task == "tense":
        if args.task_label == "past":
            config.neg_train = f"data/{dataset}/{args.task}/{args.sample_n}.present"
        elif args.task_label == "present":
            config.neg_train = f"data/{dataset}/{args.task}/{args.sample_n}.past"
        else:
            raise NotImplementedError
    else:
        config.neg_train = f"data/{dataset}/{args.task}/{args.sample_n}.{args.task_label}_neg"

    config.test = f"data/{dataset}/{args.task}/label_text100.txt"
    config.lr = args.lr ## 3e-4
    config.epoch=args.plugin_train_epochs ## 12
    config.dim_target_kl=args.dim_target_kl ## 0.1
    config.gen_k = args.gen_k ## 300
    config.dim_bottle=args.ppvae_dim_bottle ## 25
    config.relax=args.ppvae_loss_relax ## 10.0

    tokenizer = BartTokenizer.from_pretrained(args.bart_version)
    model = BartForConditionalGeneration.from_pretrained(args.bart_version)

    vae = VAE(model.model.encoder, model.model.decoder, tokenizer, config, device)
    state = torch.load(f"checkpoints/{dataset}/BART/best_val_vae_hidden.pt")
    if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
        state_copy = copy.copy(state)
        keys = state_copy.keys()
        for k in keys:
            state[k.replace('module.', '')] = state.pop(k)
    vae.load_state_dict(state)
    del state
    print("Finish Loading Pre-trained Weights..")

    model = PPVAE(vae, config, device)
    model.to(device)

    traindata = read_txt(config.train)
    negtraindata = read_txt(config.neg_train)
    trainloader = DataLoader(traindata, batch_size=args.train_batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, shuffle=True)
    neg_trainloader = DataLoader(negtraindata, batch_size=args.train_batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config.lr, correct_bias=True)
    os.makedirs(f"bart_result/ppvae/results/{dataset}", exist_ok=True)
    log_file = f"bart_result/ppvae/results/{dataset}/{args.task}-{args.task_label}-epoch{config.epoch}-bs{args.train_batch_size}-lr{config.lr}-ns{args.sample_n}.log"
    logger = Logger(log_file)

    model.train()
    best_val_loss = 99999.
    total_iters = 0
    for e in range(config.epoch):
        losses = []
        losses_rec = []
        losses_kl = []
        losses_z_rec=[]
        for batch_id, x in enumerate(tqdm(trainloader)):
            out = tokenizer.batch_encode_plus(x, return_tensors="pt", padding=True)
            pad_token_id = tokenizer.pad_token_id
            y = out['input_ids']
            y_ids = y[:, :-1].contiguous()
            y_mask = out['attention_mask'][:, :-1]
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            
            neg_x = next(iter(neg_trainloader))
            out_neg = tokenizer.batch_encode_plus(neg_x, return_tensors="pt", padding=True)
            neg_y = out_neg['input_ids']
            neg_y_ids = neg_y[:, :-1].contiguous()
            neg_y_mask = out_neg['attention_mask'][:, :-1]
            neg_lm_labels = neg_y[:, 1:].clone()
            neg_lm_labels[neg_y[:, 1:] == pad_token_id] = -100
            
            loss_z_rec, loss_rec, loss_kl, loss = model(out['input_ids'].to(device),
                                            lm_labels.to(device), 
                                            out['attention_mask'].to(device), 
                                            y_ids.to(device))
            neg_loss_z_rec, neg_loss_rec, neg_loss_kl, neg_loss = model(out_neg['input_ids'].to(device),
                                                        neg_lm_labels.to(device), 
                                                        out_neg['attention_mask'].to(device), 
                                                        neg_y_ids.to(device))
            
            loss_z_rec, loss_rec, loss_kl, loss = loss_z_rec.mean(), loss_rec.mean(), loss_kl.mean(), loss.mean()
            neg_loss_z_rec, neg_loss_rec, neg_loss_kl, neg_loss = neg_loss_z_rec.mean(), neg_loss_rec.mean(), neg_loss_kl.mean(), neg_loss.mean()
            loss = torch.max(torch.tensor(0).to(device), loss - config.alpha*neg_loss + config.relax)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().numpy())
            losses_z_rec.append(loss_z_rec.detach().cpu().numpy())
            losses_rec.append(loss_rec.detach().cpu().numpy())
            losses_kl.append(loss_kl.detach().cpu().numpy())
            
        logger.info("Train Loss       : {:.4f}".format(np.mean(losses)))
        logger.info("Train Loss z Rec : {:.4f}".format(np.mean(losses_z_rec)))
        logger.info("Train Loss Rec   : {:.4f}".format(np.mean(losses_rec)))
        logger.info("Train Loss KL.   : {:.4f}".format(np.mean(losses_kl)))
        total_iters += 1

    ## Generation
    os.makedirs(f"bart_result/ppvae/sentences/{dataset}/{args.task}", exist_ok=True)
    model.eval()
    finalsents = []
    for _ in tqdm(range(config.gen_k)):
        z = torch.randn(args.gen_batch_size, config.dim_z).to(device)
        with torch.no_grad():
            sents = model.generate(z)
        texts = []
        for ii in sents:
            endindex = (ii == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(endindex) !=0:
                texts.append(tokenizer.decode(ii[1: min(endindex)]))
            else:
                continue
        finalsents.extend(texts)
    with open(f"bart_result/ppvae/sentences/{dataset}/{args.task}/{args.task_label}-epoch{config.epoch}-bs{args.train_batch_size}-lr{config.lr}-ns{args.sample_n}-{config.gen_k}K-alpha{config.alpha}.txt", "w") as f:
        for sent in finalsents:
            f.write(sent + "\n")
    f.close()

def PCAE_plugin_training(args):
    gpu = not args.no_gpu
    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size
    device = torch.device(args.gpu[0] if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    dataset = task_dataset_dict[args.task]

    ## setup config for each training task
    class config():
        pass
    config.fb_mode=args.fb_mode ## 1
    config.dim_target_kl=args.dim_target_kl ## 0.1
    config.zmanner=args.zmanner ## "hidden"
    config.dim_label = args.dim_label ## 8
    config.dim_z=args.dim_z ## 128
    config.class_num = class_num_dataset_dict[args.task]
    config.train = f"data/{dataset}/{args.task}/label_text{args.sample_n}.txt"
    config.layer_num=args.layer_num
    config.lr = args.lr ## 4e-4
    config.epoch=args.plugin_train_epochs ## 28
    config.gen_k = args.gen_k ## 100

    tokenizer = BartTokenizer.from_pretrained(args.bart_version)
    model = BartForConditionalGeneration.from_pretrained(args.bart_version)

    vae = VAE(model.model.encoder, model.model.decoder, tokenizer, config, device)
    state = torch.load(f"checkpoints/{dataset}/BART/best_val_vae_hidden.pt")
    if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
        state_copy = copy.copy(state)
        keys = state_copy.keys()
        for k in keys:
            state[k.replace('module.', '')] = state.pop(k)
    vae.load_state_dict(state)
    del state
    print("Finish Loading Pre-trained Weights..")

    model = PCAE(vae, config, device, layer_num=config.layer_num)
    model.to(device)
    traindata = ConditionalGenerationDataset.from_file(file_path=config.train)
    trainloader = DataLoader(traindata, batch_size=args.train_batch_size, pin_memory=True, drop_last=False, num_workers=args.workers, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config.lr, correct_bias=True)
    os.makedirs(f"bart_result/pcae/results/{dataset}", exist_ok=True)
    log_file = f"bart_result/pcae/results/{dataset}/{args.task}-epoch{config.epoch}-bs{args.train_batch_size}-lr{config.lr}-ln{config.layer_num}-ns{args.sample_n}-mean{args.use_mean}.log"
    logger = Logger(log_file)

    model.train()
    best_val_loss = 99999.
    total_iters = 0
    for e in range(config.epoch):
        losses = []
        losses_rec = []
        losses_kl = []
        for batch_id, (x, ylabel) in enumerate(tqdm(trainloader)):
            out = tokenizer.batch_encode_plus(x, return_tensors="pt", padding=True)
            pad_token_id = tokenizer.pad_token_id
            y = out['input_ids']
            y_ids = y[:, :-1].contiguous()
            y_mask = out['attention_mask'][:, :-1]
            lm_labels = y[:, 1:].clone()
            lm_labels[y[:, 1:] == pad_token_id] = -100
            
            ylabel = torch.tensor(ylabel, device=device)
            loss_rec, loss_kl, loss = model(out['input_ids'].to(device), ylabel, 
                                            lm_labels.to(device), 
                                            out['attention_mask'].to(device), 
                                            y_ids.to(device), args.use_mean, beta=args.beta)
            
            loss_rec, loss_kl, loss = loss_rec.mean(), loss_kl.mean(), loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().numpy())
            losses_rec.append(loss_rec.detach().cpu().numpy())
            losses_kl.append(loss_kl.detach().cpu().numpy())
            
        logger.info("Train Loss     : {:.4f}".format(np.mean(losses)))
        logger.info("Train Loss Rec : {:.4f}".format(np.mean(losses_rec)))
        logger.info("Train Loss KL. : {:.4f}".format(np.mean(losses_kl)))
        total_iters += 1

    ## generation
    os.makedirs(f"bart_result/pcae/sentences/{dataset}/{args.task}", exist_ok=True)
    for y in range(config.class_num):
        model.eval()
        finalsents = []
        ## iteratively generate controllable texts
        for _ in tqdm(range(config.gen_k)):
            z = torch.randn(args.gen_batch_size, config.dim_z).to(device)
            with torch.no_grad():
                sents = model.generate(z, y)
            texts = []
            for ii in sents:
                endindex = (ii == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(endindex) !=0:
                    texts.append(tokenizer.decode(ii[1: min(endindex)]))
                else:
                    texts.append(tokenizer.decode(ii[1: ]))
            finalsents.extend(texts)
        with open(f"bart_result/pcae/sentences/{dataset}/{args.task}/{y}-epoch{config.epoch}-bs{args.train_batch_size}-lr{config.lr}-ln{config.layer_num}-ns{args.sample_n}-{config.gen_k}K.txt", "w") as f:
            for sent in finalsents:
                f.write(sent + "\n")
        f.close()

if __name__=="__main__":
    args = parser.parse_args()
    if args.run_mode == "vae_ft":
        VAE_finetuning(args)
    elif args.run_mode == "pcae":
        PCAE_plugin_training(args)
    elif args.run_mode == "ppvae":
        PPVAE_plugin_training(args)
    elif args.run_mode == "optimus":
        Optimus_plugin_fintuning(args)
    else:
        raise NotImplementedError