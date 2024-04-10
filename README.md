# Modality and Topological Invariant Representation Learning for User Identity Linkage

### Abstract

Vision transformers have significantly advanced the field of computer vision in recent years. 
The cornerstone of vision transformers is the multi-head attention mechanism, which models the interactions between the visual elements within a feature map. 
However, the vanilla multi-head attention paradigm learns the parameters of different heads independently and separately. 
The crucial interactions across different attention heads are ignored, leading to the redundancy and under-utilization of the model's capacity. 
In order to facilitate the model expressiveness, we propose a novel nested attention mechanism Ne-Att to explicitly model cross-head interactions via a hierarchical variational distribution. 
Extensive experiments are conducted on image classification, and the experimental results demonstrate the superiority of Ne-Att.

## Getting Started

### Dataset 
Download ImageNet1K:
https://www.pudn.com/news/62c54cf6d4521643cfa84bf5.html

### Experimental Environment
#### 1)Conda 
conda create --name vit_test python=3.8

source activate vit_test
#### 2)Pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch


### Training Code
#### Example of training from scratch.

python -m torch.distributed.launch --nproc_per_node=8 train.py  ~/data/ \
--val-split=val \
-b=64 \
--img-size=224 \
--epochs=300 \
--color-jitter=0  \
--amp  \
--sched='cosine'  \
--model-ema  \
--model-ema-decay=0.995  \
--reprob=0.5  \
--smoothing=0.1  \
--min-lr=1e-7  \
--warmup-epochs=3  \
--train-interpolation=bicubic  \
--aa=v0  \
--lr=5e-3  \
--model=vit_base_patch16_224_in21k  \
--opt=sgd  \
--weight-decay=1e-4  \
--num-classes=1000  \
--log-interval=50


### Transfer Learning Code
Using the excellent timm package, the article result can be reproduced almost completely. Specifically, timm package enables to compare official pretraining of ViT, and validate the improvement in transfer learning results. This comparison also enables to show how our pretraining stabilizes transfer learning results.

An example training code on cifar100:

python train.py \
/Cifar100Folder/ \
-d=torch/CIFAR100 \
--val-split=val \
-b=64 \
--img-size=224 \
--epochs=100 \
--color-jitter=0 \
--amp \
--sched='cosine' \
--model-ema \
--model-ema-decay=0.995 \
--reprob=0.5 \
--smoothing=0.1 \
--min-lr=1e-7 \
--warmup-epochs=3 \
--train-interpolation=bicubic \
--aa=v0 \
--lr=5e-4 \
--model=vit_base_patch16_224_in21k \
--initial-checkpoint="model_best.pth.tar" \
--opt=sgd \
--weight-decay=1e-4 \
--num-classes=100 \
--log-interval=50