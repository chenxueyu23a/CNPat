import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0,1"  # This line sets the environment variable 'CUDA_VISIBLE_DEVICES' to "0,1"
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json

for ipc in [chr(ord('A')+i) for i in range(ord('H')-ord('A')+1)]:
    if 'pat'+ipc not in os.listdir('./llama_emb/'):
        os.mkdir('./llama_emb/pat'+ipc)

model_path = "./../vicuna-7b-v1.5-16k" # 使用模型，这里采用vicuna-7b-v1.5-16k，可以替换
embed_len=4096 # 基于llama2的一类模型默认embed长度4096

t = AutoTokenizer.from_pretrained(model_path)
t.pad_token = t.eos_token
m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cuda")
m.eval()

# 可以修改file_name 所属列表
# docs_t 标题文本
# docs_a 摘要文本
# docs_c 权利要求书文本
# docs_ta 标题+摘要文本 ..以此类推tc ac tac

for ipc in ["A","B","C","D","E","F","G","H"]:   
    for file_name in ['docs_a','docs_t','docs_ta']:
        with open('./CNPat/pat'+ipc+'/'+file_name+'.txt') as f:
            docs = f.readlines()
        docs=[''.join(doc.split(' ')[1:])[:-1][:1024] for doc in docs]  # 删除序号和\n

        sent_emb_mat=np.zeros((len(docs),embed_len))
        for i,doc in enumerate(docs):
            texts = [docs[i]]
            t_input = t(texts, padding=True, return_tensors="pt").to(m.device)

            with torch.no_grad():
                last_hidden_state = m(**t_input, output_hidden_states=True).hidden_states[-1]

            weights_for_non_padding = t_input.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1,device='cuda').unsqueeze(0)
            sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
            num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
            sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
            # Apply the fully connected layer to reduce the dimensionality
            #sentence_embeddings = fc(sentence_embeddings)

            #print(t_input.input_ids)
            #print(weights_for_non_padding)
            #print(num_of_none_padding_tokens)
            #print(i,sentence_embeddings.shape)
            sent_emb_mat[i]=sentence_embeddings.detach().cpu().numpy().squeeze()
            #print(sentence_embeddings)

        np.save('./attributed_data/pat'+ipc+'/'+file_name+'_emb_mat_4096.npy',sent_emb_mat)
        print('Finish',ipc,file_name)
