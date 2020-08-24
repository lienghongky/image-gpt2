# -*- coding: utf-8 -*-
"""Transformers_Image-GPT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/apeguero1/image-gpt/blob/master/Transformers_Image_GPT.ipynb

#Run Image GPT with [Transformers](https://github.com/huggingface/transformers)
by [Alfredo Peguero-Tejada](https://twitter.com/dj__ai)

# Download Image GPT
"""

# !git clone https://github.com/openai/image-gpt.git

model_sizes = ["s", "m", "l"] #small medium large, xl not available
model_size = "s"
models_dir = "./content/models"
color_clusters_dir = "./content/clusters"
bs = 8 
n_px = 32

# !python image-gpt/download.py --model {model_size} --ckpt 1000000 --clusters --download_dir {models_dir}/{model_size}
# !python image-gpt/download.py --model s --ckpt 1000000 --clusters --download_dir ./content/models/s
# !python image-gpt/download.py --clusters --download_dir {color_clusters_dir}
# python ./download.py --clusters --download_dir ./content/clusters

"""# Subclass GPT2LMHeadModel"""

# !pip install transformers

import os
import transformers
from transformers import GPT2Config
from transformers.modeling_gpt2 import GPT2Model,GPT2LMHeadModel
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

def load_tf_weights_in_image_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")

        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ) or name[-1] in ['_step']:
            logger.info("Skipping {}".format("/".join(name)))
            continue
        
        pointer = model
        if name[-1] not in ["wtet"]:
          pointer = getattr(pointer, "transformer")
        
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            elif scope_names[0] in ['q_proj','k_proj','v_proj']:
                pointer = getattr(pointer, 'c_attn')
                pointer = getattr(pointer, 'weight')
            elif len(name) ==3 and name[1]=="attn" and scope_names[0]=="c_proj":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="wtet":
                pointer = getattr(pointer, "lm_head")
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="sos":
                pointer = getattr(pointer,"wte")
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if len(name) > 1 and name[1]=="attn" or name[-1]=="wtet" or name[-1]=="sos" or name[-1]=="wte":
           pass #array is used to initialize only part of the pointer so sizes won't match
        else:
          try:
              assert pointer.shape == array.shape
          except AssertionError as e:
              e.args += (pointer.shape, array.shape)
              raise
          
        logger.info("Initialize PyTorch weight {}".format(name))

        if name[-1]=="q_proj":
          pointer.data[:,:config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif name[-1]=="k_proj":
          pointer.data[:,config.n_embd:2*config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif name[-1]=="v_proj":
          pointer.data[:,2*config.n_embd:] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif (len(name) ==3 and name[1]=="attn" and name[2]=="c_proj" ):
          pointer.data = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) )
        elif name[-1]=="wtet":
          pointer.data = torch.from_numpy(array)
        elif name[-1]=="wte":
          pointer.data[:config.vocab_size-1,:] = torch.from_numpy(array)
        elif name[-1]=="sos":
          pointer.data[-1] = torch.from_numpy(array)
        else:
          pointer.data = torch.from_numpy(array)

    return model


from torch.nn.parameter import Parameter
class ln_mod(nn.Module):
    def __init__(self, nx,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.Tensor(nx))
    def forward(self,x):#input is not mean centered
        return x / torch.sqrt( torch.std(x,axis=-1,unbiased=False,keepdim=True)**2 + self.eps ) * self.weight.data[...,:] 

def replace_ln(m, name,config):
  for attr_str in dir(m):
      target_attr = getattr(m, attr_str)
      if type(target_attr) == torch.nn.LayerNorm:
          #print('replaced: ', name, attr_str)
          setattr(m, attr_str, ln_mod(config.n_embd,config.layer_norm_epsilon))

  for n, ch in m.named_children():
      replace_ln(ch, n,config)        

def gelu2(x):
    return x * torch.sigmoid(1.702 * x)

class ImageGPT2LMHeadModel(GPT2LMHeadModel):
  load_tf_weights = load_tf_weights_in_image_gpt2
  
  def __init__(self, config):
      super().__init__(config)
      self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
      replace_ln(self,"net",config) #replace layer normalization
      for n in range(config.n_layer):
        self.transformer.h[n].mlp.act = gelu2 #replace activation 

  def tie_weights(self): #image-gpt doesn't tie output and input embeddings
    pass

"""# Unconditional Image Generation"""

import numpy as np
color_clusters_file = "%s/kmeans_centers.npy"%(color_clusters_dir)
clusters = np.load(color_clusters_file) #get color clusters

MODELS={"l":(1536,16,48),"m":(1024,8,36),"st":(1024,8,36),"s":(512,8,24) } 
n_embd,n_head,n_layer=MODELS[model_size] #set model hyperparameters
vocab_size = len(clusters) + 1 #add one for start of sentence token
print('vocab_size',vocab_size)
print('vocab_size= ',vocab_size,'n_ctx= ',n_px*n_px,'n_positions= ',n_px*n_px,'n_embd= ',n_embd,'n_layer= ',n_layer,'n_head= ',n_head)
config=GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_px*n_px,
        n_ctx=n_px*n_px,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        bos_token_id=50256,
        eos_token_id=50256
    )
config.vocab_size = vocab_size
# config = transformers.GPT2Config(vocab_size=vocab_size,n_ctx=n_px*n_px,n_positions=n_px*n_px,n_embd=n_embd,n_layer=n_layer,n_head=n_head)
model_path = "%s/%s/model.ckpt-1000000.index"%(models_dir,model_size)
print('model config ',config)
model = ImageGPT2LMHeadModel.from_pretrained(model_path,from_tf=True,config=config)
print('Model loaded',type(model))

context = np.full( (bs,1), vocab_size - 1 ) #initialize with SOS token
context = torch.tensor(context)
# output = model.generate(input_ids=context,pad_token_id=50256,max_length= n_px*n_px + 1,temperature=1.0,do_sample=True,top_k=40)
# output = model(input_ids=context)

# print('done ouput model',type(output))
# Commented out IPython magic to ensure Python compatibility.
#visualize samples with Image-GPT color palette.
# %matplotlib inline
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# print('output ',type(model))

# samples = np.asarray(output[:,1:])

# print('Len',samples.shape)
# f, axes = plt.subplots(1,bs,dpi=300)
# samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color cluster tokens back to pixels
# f, axes = plt.subplots(1,bs,dpi=300)

# for img,ax in zip(samples_img,axes):
#     # ax.axis('off')
#     ax.imshow(img)
#     print('showing cluster .. ')

"""# Tokenize Cropped Images for Image Completion"""

#numpy implementation of functions in image-gpt/src/utils which convert pixels of image to nearest color cluster. 
def normalize_img(img):
  return img/127.5 - 1

def squared_euclidean_distance_np(a,b):
  b = b.T
  a2 = np.sum(np.square(a),axis=1)
  b2 = np.sum(np.square(b),axis=0)
  ab = np.matmul(a,b)
  d = a2[:,None] - 2*ab + b2[None,:]
  return d

def color_quantize_np(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    return np.argmin(d,axis=1)

#get images
# curl https://i.imgur.com/fIiwqyn.jpeg > sg.jpeg
imgs = ["vr.png","pst.png","ps.png","lh.png","cat.jpeg","sg.jpeg"]


#Resize original images to n_px by n_px
import cv2
import numpy as np
dim=(n_px,n_px)
for imgPath in imgs:
    image_paths = [imgPath]*bs
    x = np.zeros((bs,n_px,n_px,3),dtype=np.uint8)

    for n,image_path in enumerate(image_paths):
      img_np = cv2.imread(image_path)   # reads an image in the BGR format
      img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)   # BGR -> RGB
      H,W,C = img_np.shape
      D = min(H,W)
      img_np = img_np[:D,:D,:C] #get square piece of image
      x[n] = cv2.resize(img_np,dim, interpolation = cv2.INTER_AREA) #resize to n_px by n_px



    #use Image-GPT color palette and crop images
    x_norm = normalize_img(x) #normalize pixels values to -1 to +1
    samples = color_quantize_np(x_norm,clusters).reshape(x_norm.shape[:-1]) #map pixels to closest color cluster

    n_px_crop = 16
    primers = samples.reshape(-1,n_px*n_px)[:,:n_px_crop*n_px] # crop top n_px_crop rows. These will be the conditioning tokens

    #visualize samples and crops with Image-GPT color palette. Should look similar to original resized images
    samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color clusters back to pixels
    primers_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px_crop,n_px, 3]).astype(np.uint8) for s in primers] # convert color clusters back to pixels


    # f, (ax1, ax2, ax3) = plt.subplots(3,bs,dpi=300)
    # for img,ax in zip(x,ax1):
    #     ax.axis('off')
    #     ax.imshow(img)
    # for img,ax in zip(samples_img,ax2):
    #     ax.axis('off')
    #     ax.imshow(img)
    # for img,ax in zip(primers_img,ax3):
    #     ax.axis('off')
    #     ax.imshow(img)
    # try:
    #     fname = 'output_'+imgPath
    #     plt.savefig(fname)
    #     plt.close(f)
    # except Exception:
    #     plt.show()

    print('croped images done!')
    """# Conditional Image Completion"""

    context = np.concatenate( (np.full( (bs,1), vocab_size - 1 ),primers,), axis=1 )
    context = torch.tensor(context)
    print('Generating . . .')
    output = model.generate(input_ids=context,pad_token_id=50256,max_length= n_px*n_px + 1,temperature=1.0,do_sample=True,top_k=40)
    # output = model.generate(input_ids=context)
    print('Generate Done!')
    #visualize samples with Image-GPT color palette.    

    samples = output[:,1:].cpu().detach().numpy()
    samples_img_final = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color cluster tokens back to pixels


    f, (ax1, ax2, ax3, ax4) = plt.subplots(4,bs,dpi=300)
    for img,ax in zip(x,ax1):
        ax.axis('off')
        ax.imshow(img)
    for img,ax in zip(samples_img,ax2):
        ax.axis('off')
        ax.imshow(img)
    for img,ax in zip(primers_img,ax3):
        ax.axis('off')
        ax.imshow(img)
    for img,ax in zip(samples_img_final,ax4):
        ax.axis('off')
        ax.imshow(img)
    try:
        fname = 'output_'+imgPath
        plt.savefig(fname)
        plt.close(f)
    except Exception:
        plt.show()
    print('done!')