import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from model.tokenizer import build_tokenizer
from rdkit import Chem
import ast

def pre_processing_smi(smiles_str, tokenizer,smiles_maxlen=300):
    encode_dict = tokenizer.encode_plus(
        text=smiles_str,  # 输入的smiles字符串
        max_length=smiles_maxlen,  # 300
        padding='max_length',  # 300
        truncation=True)  # 超过最大长度截断处理
    smiles_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).view(1, -1)  # smiles ----> tensor
    smiles_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).view(1, -1)
    return smiles_ids, smiles_mask

def pre_processing_smi(smiles_str, tokenizer, config_json):
    smiles_str = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_str))
    encode_dict = tokenizer.encode_plus(
        text=smiles_str,  
        max_length=config_json["dataset_config"]["smiles_maxlen"], 
        padding='max_length', 
        truncation=True)  
    smiles_ids = torch.from_numpy(
        np.array(encode_dict['input_ids'])).view(1, -1)
    smiles_mask = torch.from_numpy(
        np.array(encode_dict['attention_mask'])).view(1, -1)
    return [smiles_ids, smiles_mask]

def pre_processing_smi(smiles_str, tokenizer, config_json):
    if isinstance(smiles_str, str):  # 判断smiles_str是否为字符串类型
        encode_dict = tokenizer.encode_plus(
            text=smiles_str,  # 输入的smiles字符串
            max_length=config_json["dataset_config"]["smiles_maxlen"],  # 300
            padding='max_length',  # 300
            truncation=True)  # 超过最大长度截断处理
        smiles_ids = torch.from_numpy(
            np.array(encode_dict['input_ids'])).view(
            1, -1)  # smiles ----> tensor
        smiles_mask = torch.from_numpy(
            np.array(encode_dict['attention_mask'])).view(1, -1)
    else:
        smiles_ids = []
        smiles_mask = []
        for i in smiles_str:
            encode_dict = build_tokenizer.encode_plus(
                text=i,
                max_length=config_json["dataset_config"]
                ["smiles_maxlen"],
                padding='max_length',
                truncation=True)
            smiles_ids.append(
                torch.tensor(encode_dict['input_ids']))
            smiles_mask.append(
                torch.tensor(encode_dict['attention_mask']))
        smiles_ids = torch.stack(smiles_ids)
        smiles_mask = torch.stack(smiles_mask)
    return [smiles_ids, smiles_mask]