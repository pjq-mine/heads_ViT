# Modality and Topological Invariant Representation Learning for User Identity Linkage

### Abstract

  The integration of heterogeneous multi-modal posts and social topological structure across various social media platforms is crucial for accomplishing the task of user identity linkage.
  Mainstream approaches addressing this task have developed sophisticated architectures and techniques for multi-modality fusion or graph structural representation learning. 
  However, the natural heterogeneity of multi-modal posts and the diversity of social topological connections across platforms lead to semantic gaps and distribution shifts that contain task-irrelevant information.
  In this paper, we aim to learn effective Modality and Topological Invariant representations to facilitate the process of user identity linkage.
  We propose a novel framework, MoToInv, which marries the learning of invariant representations with the dual insights drawn from multi-modal user posts and the underlying social network connections.
  Specifically, we propose an adversarial network to disentangle the modality-invariant and -variant representation, which captures the commonality and specificity across modalities to ease the subsequent fine-grained multi-modal fusion process.
  Moreover, in the topological invariant learning stage, we design a method based on the invariant subgraph extractor. 
  This method utilizes two mixup strategies for invariant and variant subgraphs, aiming to better capture invariant topological patterns and eliminate spurious social correlations.
  Experiments conducted on real world multi-modal user identity linkage dataset, show that our proposed framework significantly outperforms the most advanced methods.

## Getting Started


### Experimental Environment
#### 1)Conda 
```python
conda create --name MoToInv python=3.8
```
```python
source activate MoToInv
```

#### 2)Tensorflow 
```python
conda install tensorflow-gpu==1.10
```

#### 3)sklearn 
```python
conda install -c anaconda scikit-learn
```

### Training Code
```python
python main.py
```