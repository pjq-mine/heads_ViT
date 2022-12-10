# Rich Attention Heads Interaction in Vision Transformer

### Abstract

Transformers have achieved great success in the field of natural language processing (NLP) due to the multi-head attention (MHA) mechanism. Inspired by this, more and more scholars introduce Transformers into computer vision to capture global information of images, achieving competitive performance to Convolutional Neural Networks (CNNs). However, many studies have found redundancy between the heads of the MHA due to the lack of interaction, which affects the performance of the model. Thanks to the explaining-away effects in the probabilistic graphical model, it becomes possible to improve the rich interaction among different heads. With the help of the hierarchical structure of the transformer, we pass head attention values layer by layer for fusion, which effectively improves the independence between heads. We conduct extensive experiments and demonstrate that our proposed model outperforms vanilla Vision Transformer (ViT), by 1.56% top1 accuracy on ImageNet1K in image classification,  2.66-3.47% improvement on downstream datasets of transfer learning, due to its improvements on the parameter efficiency and heads diversity.

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