# TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition
## Abstract
  Going beyond few-shot action recognition (FSAR), cross-domain FSAR (CDFSAR) has attracted recent research interests by solving the domain gap lying in source-to-target transfer learning. Existing CDFSAR methods mainly focus on joint training of source and target data to mitigate the side effect of domain gap. However, such kind of methods suffer from two limitations: First, pair-wise joint training requires retraining deep models in case of one source data and multiple target ones, which incurs heavy computation cost, especially for large source and small target data. Second, pre-trained models after joint training are adopted to target domain in a straightforward manner, hardly taking full potential of pre-trained models and then limiting recognition performance. To overcome above limitations, this paper proposes a simple yet effective baseline, namely Temporal-Aware Model Tuning (TAMT) for CDFSAR. Specifically, our TAMT involves a decoupled paradigm by performing pre-training on source data and fine-tuning target data, which avoids retraining for multiple target data with single source. To effectively and efficiently explore the potential of pre-trained models in transferring to target domain, our TAMT proposes a Hierarchical Temporal Tuning Network (HTTN), whose core involves local temporal-aware adapters (TAA) and a global temporal-aware moment tuning (GTMT). Particularly, TAA learns few parameters to recalibrate the intermediate features of frozen pre-trained models, enabling efficient adaptation to target domains. Furthermore, GTMT helps to generate powerful video representations, improving match performance on the target domain. Experiments on several widely used video benchmarks show our TAMT outperforms the recently proposed counterparts by 13\%$\sim$31\%, achieving new state-of-the-art CDFSAR results.
 
  ![image](https://github.com/Dikay1/OS-AGDO/blob/main/assets/frame.jpg)
  ###### Fig.2. (a) Overview of our TAMT paradigm, which pre-trains the models on $\mathcal{S}_{CD}$ and fine-tunes them on $\mathcal{T}_{CD}$. Specifically, for pre-training stage, the model is first optimized with a reconstruction-based SSL solution, while the encoder $\mathcal{E}$ is post-trained with the SL objective. Subsequently, the pre-trained $\mathcal{E}$ is fine-tuned for few-shot adaptation on $\mathcal{T}_{CD}$ by using our  HTTN. (b) HTTN for few-shot adaptation, where a metric-based is used for few-shot adaptation. Particularly, our HTTN consists of local Temporal-Aware Adapters (TAA) and Global Temporal-aware Moment Tuning (GTMT).


## TODO

- [ ] Release the code.
- [√] Release the [arxiv preprint](https://arxiv.org/pdf/2411.19041).

## Citation
If our work is helpful to you, please consider citing us by using the following BibTeX entry:


## Usage
### 1.Requirements
  Code is tested under Pytorch 1.9.1, python 3.6.10, and CUDA 11.3 
  
```
pip install -r requirements.txt
```
### 2.Dataset
  
  Put the data the same with your filelist:  
```
hmdb51_org
├── brush_hair
└── cartwheel
```
### 3.Train and Test
  Run following commands to start training or testing:

```
cd scripts/hmdb51/run_meta_deepbdc
sh run_metatrain.sh #for train and test, for individual training or testing, please comment out parts of the code yourself.
```
 
