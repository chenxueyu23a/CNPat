import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json


print('------------------------')
with open('docs_t_emb_mat_4096.npy','rb') as f:
    vecs=np.load(f)
print(np.isinf(vecs).sum())
# 替换inf
vecs[np.isinf(vecs)]=np.nan
vecs[np.isnan(vecs)]=0
print(np.isinf(vecs).sum())

# 参考BERT-whitening
# 计算kernel和bias
def compute_kernel_bias(vecs, n_components=512):
    """
    n_components为PCA前n维特征
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

kernel, bias = compute_kernel_bias(vecs)


# 计算相似度
def transform_and_normalize(vecs, kernel=None, bias=None):
    """
    应用变换，然后标准化
    最后的变换：y = (x + bias).dot(kernel)
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def compute_cosine(a, b, kernel_=None, bias_=None):
    a_vec=transform_and_normalize(a, kernel_, bias_)
    b_vec=transform_and_normalize(b, kernel_, bias_)
    return round((a_vec * b_vec).sum(axis=1).tolist()[0], 2)
    
from scipy.stats import spearmanr
def compute_corrcoef(x, y):
    """使用Spearman相关系数进行评估"""
    return spearmanr(x, y).correlation


