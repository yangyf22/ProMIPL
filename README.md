# A PyTorch Implementation of ProMIPL

This is a PyTorch implementation of our paper "PROMIPL: A Generative Probabilistic Disambiguation Model for Multi-Instance Partial-Label Learning"


## Requirements

the file `environment.yml` records the requirement packages for this project.

To install the requirement packages, please run the following command:

```sh
conda env create -f environment.yml -n MIPL
```

Then, the environment can be activated by using the command

```sh
conda activate MIPL
```


## Datasets

The datasets used in this paper can be found on this [link](http://palm.seu.edu.cn/zhangml/Resources.htm#MIPL_data).



## Demo

To reproduce the results of MNIST_MIPL dataset in the paper, please run the following command:

```sh
python main.py --ds MNIST --ds_suffix r1 --decay_tau 0.0 --lr 0.005
python main.py --ds MNIST --ds_suffix r2 --decay_tau 0.0 --lr 0.005
python main.py --ds MNIST --ds_suffix r3 --decay_tau 0.0 --lr 0.0085
```
