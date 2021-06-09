# PositionBias-in-Emotion-Cause-Analysis
ACL2021:Position Bias Mitigation: A Knowledge-Aware Graph Model for EmotionCause Extraction

[Hanqi Yan](https://github.com/hanqi-qi), [Lin Gui](https://warwick.ac.uk/fac/sci/dcs/people/lin_gui/), [Gabriele Pergola](https://warwick.ac.uk/fac/sci/dcs/people/u1898418/), [Yulan He](https://warwick.ac.uk/fac/sci/dcs/people/yulan_he/).

In this work, we find that a widely-used ECE dataset exhibits a poistion bias and existing models tend to rely on the relative information and suffer from the dataset bias. Our proposed knowledge-aware model performs on par with the existing methods on the original ECE dataset, and is more robust against adversarial samples whose relative imformation has been changed. [Our paper contains further details](https://arxiv.org/abs/2103.03404). This repository contains the code for our experiments.

<p>
<img src="model_view.pdf"  width="550" >
</p>

## Requirements

To install a working environment:

```
conda create --name rank-collapse python=3.8
conda activate rank-collapse
pip install git+git://github.com/huggingface/transformers.git@12d7624199e727f37bef7f53d527df7fabdb1fd6
conda install pytorch torchvision torchaudio -c pytorch
conda install -c anaconda scipy scikit-learn black isort matplotlib flake8
```

## Code Structure

This repo contains three parts to reproduce our experiments, i.e., extract knowledge paths from the ConceptNet, incorporat the knowledge paths to capturing the causal relations between the document clauses, generate adversarial samples and evaluate existing ECE models on these sampels.  

### Path Extraction
We first extract knowledeg paths which contain less than two intermediate entities from ConceptNet. [This  publicly released package](https://github.com/INK-USC/KagNet) contains the driver code to extract all the knowledge paths between the given head entity and the tail entity. Our code provides how to identity the keywords as the head/tail entity, and the path filter mechanism.

### Knowledge-aware graph model
This part use the extracted paths to identity the cause clauses in a document:
```
python run_sort.py --width 2 --depth 6 --hidden_dim 48 --seed 2 --num_labels 10 --seq_len 8 --n_epochs 65 --path_len 0 --n_paths 5 --n_train_data 1000 --n_repeat 5 --n_eval_data 200 --no_sub_path
```

### Adversarial Attacks
This part genetrate the 
```
python run_convex_hull.py --width 3 --depth 6 --seq_len 10 --hidden_dim 84 --seed 2 --num_labels 2 --n_epochs 70 --path_len 0 --n_paths 5 --n_train_data 10000 --n_repeat 5 --ffn2 --n_eval_data 300 --no_sub_path
```

## Citation

If you find our work useful, please cite as:

```
@article{rankCollapse2021,
  title         = {Attention is not all you need, pure attention loses rank doubly exponentially with depth},
  author        = {Dong, Yihe and Cordonnier, Jean-Baptiste and Loukas, Andreas},
  url       	= {https://arxiv.org/abs/2103.03404},
  year          = {2021}
  }
```