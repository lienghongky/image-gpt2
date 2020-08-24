#  IMAGE-GPT image conpletion


![Program](/output_ps.png)

[pyversion-button]: https://img.shields.io/pypi/pyversions/Markdown.svg
[![Just take it!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/AtsushiSakai)

IMAGE-GPT  Using gpt2 which is the large transformer model trained on language, originaly trained for text completion to generate coherent text, the same exact model trained on pixel sequences can generate coherent image completions and samples.

### what the code in this repo does :

  - Load GPT2 Model
  - Loop the input images array ex.['sg.jpeg','sg.jpeg']
  - Every single image resize it and crop haft top of it.
  - Let the model predict the missing pixel and complete it
  - plot all input and output on a figure :
  
  ![Program](/output_cat.png) 

# New Features!

  - It is just part of my testing and I don't mean to put it for public use, but I thought maybe it could help someone who like to try a working piece of code and start from there.
  - Bad code ? Yes I know! just get through it.



  
### Installation

This project requires python 3.6+

Install the dependencies.
```sh
$ conda create --name image-gpt python=3.7.3
$ conda activate image-gpt

$ conda install numpy=1.16.3
$ conda install tensorflow-gpu=1.13.1

$ conda install imageio=2.8.0
$ conda install requests=2.21.0
$ conda install tqdm=4.46.0

//NOTE it could be as simple as pip install transformers but there\'s some error so I got it to work by using : 
$ pip install -e git+https://github.com/huggingface/transformers.git@master#egg=transformers
```

### To run the program
```sh
$ python ./download.py --model s --ckpt 1000000 --clusters --download_dir ./content/models/s
$ python ./download.py --clusters --download_dir ./content/clusters

```
```sh
$ conda activate image-gpt
$ python python transformers_image_gpt.py 
```

### Todos

 - read the bad code
 - install libs and calm down.
 - drink coffee if it takes to long to install libs

License
----

@[Lieng Hongky]


**Free Software, Hell Yeah!**

   [git-repo-url]: <https://github.com/lienghongky/image-gpt2.git>
   [Lieng Hongky]: <https://github.com/lienghongky>
 





# ORIGINAL DOC
**Status:** Archive (code is provided as-is, no updates expected)

# image-gpt

Code and models from the paper ["Generative Pretraining from Pixels"](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf).

Supported Platforms:

- Ubuntu 16.04

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html, or install the dependencies shown below manually.

```
conda create --name image-gpt python=3.7.3
conda activate image-gpt

conda install numpy=1.16.3
conda install tensorflow-gpu=1.13.1

conda install imageio=2.8.0
conda install requests=2.21.0
conda install tqdm=4.46.0
```

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with image GPT (iGPT). Our code forks GPT-2 to highlight that it can be easily applied across domains. The diff from `gpt-2/src/model.py` to `image-gpt/src/model.py` includes a new activation function, renaming of several variables, and the introduction of a start-of-sequence token, none of which change the model architecture.

### Downloading Pre-trained Models

To download a model checkpoint, run `download.py`. The `--model` argument should be one of "s", "m", or "l", and the `--ckpt` argument should be one of "131000", "262000", "524000", or "1000000".

```
python download.py --model s --ckpt 1000000
```

This command downloads the iGPT-S checkpoint at 1M training iterations. The default download directory is set to `/root/downloads/`, and can be changed using the `--download_dir` argument.

### Downloading Datasets

To download datasets, run `download.py` with the `--dataset` argument set to "imagenet" or "cifar10".

```
python download.py --model s --ckpt 1000000 --dataset imagenet
```

This command additionally downloads 32x32 ImageNet encoded with the 9-bit color palette described in the paper. The datasets we provide are center-cropped images intended for evaluation; random cropped images are required to faithfully replicate training.

### Downloading Color Clusters

To download the color cluster file defining our 9-bit color palette, run `download.py` with the `--clusters` flag set.

```
python download.py --model s --ckpt 1000000 --dataset imagenet --clusters
```

This command additionally downloads the color cluster file. `src/run.py:sample` shows how to decode from 9-bit color to RGB and `src/utils.py:color_quantize` shows how to go the other way around.

### Sampling

Once the desired checkpoint and color cluster file are downloaded, we can run the script in sampling mode. The following commands sample from iGPT-S, iGPT-M, and iGPT-L respectively:

```
python src/run.py --sample --n_embd 512  --n_head 8  --n_layer 24
python src/run.py --sample --n_embd 1024 --n_head 8  --n_layer 36
python src/run.py --sample --n_embd 1536 --n_head 16 --n_layer 48
```

If your data is not in `/root/downloads/`, set `--ckpt_path` and `--color_cluster_path` manually. To run on fewer than 8 GPUs, use a command of the following form:

```
CUDA_VISIBLE_DEVICES=0,1 python src/run.py --sample --n_embd 512  --n_head 8  --n_layer 24 --n_gpu 2
```

### Evaluating

Once the desired checkpoint and evaluation dataset are downloaded, we can run the script in evaluation mode. The following commands evaluate iGPT-S, iGPT-M, and iGPT-L on ImageNet respectively:

```
python src/run.py --eval --n_embd 512  --n_head 8  --n_layer 24
python src/run.py --eval --n_embd 1024 --n_head 8  --n_layer 36
python src/run.py --eval --n_embd 1536 --n_head 16 --n_layer 48
```

If your data is not in `/root/downloads/`, set `--ckpt_path` and `--data_path` manually. You should see that the test generative losses are 2.0895, 2.0614, and 2.0466, matching Figure 3 in the paper.

### Citation

Please use the following bibtex entry:
```
@article{chen2020generative,
  title={Generative Pretraining from Pixels},
  author={Chen, Mark and Radford, Alec and Child, Rewon and Wu, Jeff and Jun, Heewoo and Dhariwal, Prafulla and Luan, David and Sutskever, Ilya},
  year={2020}
}
```

## License

[Modified MIT](./LICENSE)
# image-gpt2
