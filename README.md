## PCAE: A Framework of Plug-in Conditional Auto-Encoder for Controllable Text Generation

Official PyTorch implementation of *[PCAE: A Framework of Plug-in Conditional Auto-Encoder for Controllable Text Generation](https://www.sciencedirect.com/science/article/pii/S0950705122008942)*, published in *Knowledge-Based Systems*. We provide PCAE as well as all implemented baselines (PPVAE and Optimus) under pre-trained BART.

- [2022-9-27] Our paper is now in this [paper list](https://github.com/ImKeTT/CTG-latentAEs), which aims at collecting all kinds of controllable text generation methods using latent variational auto-encoder. Feel free to check it out and contribute!

![pcae_struct](pics/pcae_struct.jpg)



### Setup

Make sure you have installed

```markdown
transformers
tqdm
torch
numpy
```

### Dataset

We conduct five tasks span from three datasets: *Yelp review*, *Titles* and *Yahoo Question*. 

We provide our full processed datasets in [BaiduPan]() with password `i`. Please download the `data` folder and replace the empty `data` folder with it in current directory.

### Training

#### Stage 1 BART VAE Finetuning

Finetuning on three datasets. (choose DATA from `yelp`, `yahoo`, `titles`, and EPOCH from 8, 10, 10):

```bash
DATA=yelp
EPOCH=8
python train.py --run_mode vae_ft --dataset $DATA --zmanner hidden\
				--gpu 0 1 --dim_z 128 --per_gpu_train_batch_size 64\
        --train_epochs $EPOCH --fb_mode 1 --lr 5e-4
```

#### Stage 2.1 PCAE Plug-in Training

Plug-in training of PCAE. Choose arguments follow options below:

+ TASK: [sentiment,  tense,  topics, quess_s, quess]
+ SAMPLE_N: [100, 300, 500, 800, 1000]
+ NUM_LAYER: int number from 8 to 15 is fine
+ EPOCH: 10 to 20 is fine, less SAMPLE_N means less EPOCH required

```bash
TASK=sentiment
EPOCH=15
SAMPLE_N=100
NUM_LAYER=10

python train.py --run_mode pcae --task $TASK --zmanner hidden\
				--gpu 0 --dim_z 128 --per_gpu_train_batch_size 64\
        --plugin_train_epochs $EPOCH --fb_mode 1 --sample_n $SAMPLE_N\
        --layer_num $NUM_LAYER --lr 1e-4
        
python3 train.py --run_mode pcae --task sentiment --zmanner hidden --gpu 0 --dim_z 128 --per_gpu_train_batch_size 3 --plugin_train_epochs 10 --fb_mode 1 --sample_n 100 --layer_num 8 --lr 1e-4
```





#### Stage 2.2 PPVAE Plug-in Training





#### Stage 2.3 Optimus_{bart} Plug-in Finetuning



### Others

Please email me or open an issue if you have further questions.

if you find our work useful, please cite the paper and star the repo~ :)

```
@article{tu2022pcae,
  title={PCAE: A framework of plug-in conditional auto-encoder for controllable text generation},
  author={Tu, Haoqin and Yang, Zhongliang and Yang, Jinshuai and Zhang, Siyu and Huang, Yongfeng},
  journal={Knowledge-Based Systems},
  pages={109766},
  year={2022},
  publisher={Elsevier}
}
```

We thank open sourced codes related to VAEs, which inspired our work !!