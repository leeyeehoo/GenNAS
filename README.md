#  Generic Neural Architecture Search via Regression

This repository is the official implementation of Generic Neural Architecture Search via Regression ([NeurIPS'21 spotlight](https://papers.nips.cc/paper/2021/hash/aba53da2f6340a8b89dc96d09d0d0430-Abstract.html) | [Openreview](https://openreview.net/forum?id=mPTfR3Upe0o) | [Arxiv version](https://arxiv.org/abs/2108.01899)). 


## Requirements

```
pip install -r requirement.txt
```

### Minimum required datasets:

Download the data for [NDS dataset](https://dl.fbaipublicfiles.com/nds/data.zip)

```
mv <path_to_data> <path_to_GenNAS>/data/
```

Download the data for NASBench-NLP

```
git clone https://github.com/fmsnew/nas-bench-nlp-release.git
mv ./nas-bench-nlp-release/train_logs_single_run <path_to_GenNAS>/data/
mv ./nas-bench-nlp-release/train_logs_wikitext-2 <path_to_GenNAS>/data/
```
Download the data for [ImageNet16](https://drive.google.com/drive/folders/1NE63Vdo2Nia0V7LK1CdybRLjBFY72w40)
```
mv <path_to_data>/* <path_to_GenNAS>/data/ImageNet16
```
### Suggest datasets & API:
[NASBench-101](https://github.com/google-research/nasbench)
[NASBench-201](https://github.com/D-X-Y/NAS-Bench-201)

## Proxy Task Search

To search for a proxy task, run the following examples:

```
python do_search.py --search_space=nasbench101 --json_loc=data/nasbench1_search_20samples.json --json_description=nasbench1_search_20samples #NASBench-101
python do_search.py --search_space=nlp #NASBench-NLP
python do_search.py --search_space=DARTS #NDS
```


## Sampling Experiments

To do the sampling experiments, run the following examples:

```
python do_sample.py --search_space=nasbench101 --config=CONF_NB101 --json_loc=data/nasbench1_500_fb.json --json_description=nasbench1_500_fb #NASBench-101
python do_sample.py --search_space=nasbench201 --config=CONF_NB101 --json_loc=data/nasbench2_1000_0.json --json_description=nasbench2_1000_0 --dataset=cifar10 #NASBench-201
python do_sample.py --search_space=nlp --config=CONF_NLP #NASBench-NLP
python do_sample.py --search_space=DARTS --config=CONF_DARTS #NDS
```
## Exploring Experiments
To do the exploring experiments, run the following examples:
```
python do_explore.py --search_space=nasbench101 --config=CONF_NB101 #NASBench-101
python do_explore.py --search_space=nasbench201 --config=CONF_NB101 #NASBench-201
python do_explore.py --search_space=nlp --config=CONF_NLP #NASBench-NLP
python do_explore.py --search_space=DARTS --config=CONF_DARTS #NDS
```

