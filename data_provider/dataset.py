import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from model.tokenizer import build_tokenizer
from rdkit import Chem
import ast

class NIST17Dataset(Dataset):
    def __init__(self,
                 path,
                 config_json,
                 data_end=268,
                 mode='train'):
        self.mode = mode
        self.data = []
        self.tokenizer = build_tokenizer(config_json['roberta_tokenizer_path'])
        self.config_json = config_json
        self.device = self.config_json['device']

        if mode == 'train':
            for i in tqdm(range(data_end)):
                begin, end = i * 1000 + 1, (i + 1) * 1000
                filename = "/data/nist17_data/train_nist17/" + str(begin) + "-" + str(end) + ".json"
                with open(path + filename, 'r') as fp:  
                    file_data = json.load(fp)
                for tmp_data in file_data:
                    target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                    'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask]  
                    self.data.append(target_data)
            # flag = ((i % 5) in [2])
        elif mode == 'dev':
            filename = "/data/nist17_data/dev_nist17/dev.json"
            with open(path + filename, 'r') as fp:  
                file_data = json.load(fp)
            for tmp_data in file_data:
                target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask]  
                self.data.append(target_data)
            # flag = (i % 10 == 8)
        elif mode == 'test':
            filename = "/data/nist17_data/test_nist17/test.json"
            with open(path + filename, 'r') as fp:  
                file_data = json.load(fp)
            for tmp_data in file_data:
                target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask]  
                self.data.append(target_data)
        
        elif mode == 'all':
            for i in tqdm(range(data_end)):
                begin, end = i * 1000 + 1, (i + 1) * 1000
                filename = "/data/nist17_data/all_neims_uni/" + str(begin) + "-" + str(end) + ".json"
                with open(path + filename, 'r') as fp:  
                    file_data = json.load(fp)
                for tmp_data in file_data:
                    target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                    'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask]  
                    self.data.append(target_data)
        
        print('Finished reading the {} set of NIST Dataset ({} samples found)'.format(mode, len(self.data)))  

    def __getitem__(self, index):
        # Returns one sample at a time
        return self.data[index]["smiles"], self.data[index]["peaks"]

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)


class NIST23Dataset(Dataset):
    def __init__(self,
                 path,
                 config_json,
                 data_end=70,
                 mode='train'):
        self.mode = mode
        self.data = []
        self.tokenizer = build_tokenizer(config_json['roberta_tokenizer_path'])
        self.config_json = config_json
        self.device = self.config_json['device']

        if mode == 'train':
            for i in tqdm(range(data_end)):
                begin, end = i * 5000 + 1, (i + 1) * 5000
                filename = "/data/nist23_data/train23/" + str(begin) + "-" + str(end) + ".json"
                with open(path + filename, 'r') as fp:  
                    file_data = json.load(fp)
                for tmp_data in file_data:
                    target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                    'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask]  
                    self.data.append(target_data)
            
        elif mode == 'dev':
            filename = "/data/nist23_data/dev23/dev.json"
            with open(path + filename, 'r') as fp:  
                file_data = json.load(fp)
            for tmp_data in file_data:
                target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask]  
                self.data.append(target_data)

        elif mode == 'test':
            filename = "/data/nist23_data/test23/test.json"
            with open(path + filename, 'r') as fp:  
                file_data = json.load(fp)
            for tmp_data in file_data:
                target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask]  
                self.data.append(target_data)
        
        elif mode == 'all':
            for i in tqdm(range(data_end)):
                begin, end = i * 5000 + 1, (i + 1) * 5000
                filename = "/data/nist23_data/mainlib23_right_smiles_uni/" + str(begin) + "-" + str(end) + ".json"
                with open(path + filename, 'r') as fp:  
                    file_data = json.load(fp)
                for tmp_data in file_data:
                    target_data = {'smiles': pre_processing_smi(tmp_data['smiles'], self.tokenizer, config_json),   # [smiles_ids, smiles_mask]  
                                    'peaks': pre_processing_ms(tmp_data['peaks'], config_json)}  # [ms_tensor, ms_mask] 

        print('Finished reading the {} set of NIST Dataset ({} samples found)'.format(mode, len(self.data))) 

    def __getitem__(self, index):
        # Returns one sample at a time
        return self.data[index]["smiles"], self.data[index]["peaks"]

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)


def load_analysis_data(analysis_data, config_json):
    """
    Args:
        analysis_data (tuple): A tuple containing (SMILES, EIMS).
            - SMILES (list): A list of SMILES strings. e.g. ['C', 'CC', ...]
            - EIMS (list): A list of peaks data. Each item can be a list of peaks [[mz, inten], ...] or a string representation of it.
        tokenizer: RobertaTokenizer object for tokenizing SMILES.
        config_json (dict): Configuration dictionary containing 'device' and 'dataset_config'.
    """
    SMILES, EIMS = analysis_data
    # Preprocess SMILES
    smiles_ids_list = []
    smiles_mask_list = []
    tokenizer = build_tokenizer(config_json['roberta_tokenizer_path'])
    for smiles in SMILES:
        smiles_ids, smiles_mask = pre_processing_smi_with_canonical(smiles, tokenizer, config_json)
        smiles_ids_list.append(smiles_ids)
        smiles_mask_list.append(smiles_mask)

    smiles_ids = torch.cat(smiles_ids_list, dim=0).to(config_json["device"])
    smiles_mask = torch.cat(smiles_mask_list, dim=0).to(config_json["device"])

    # Preprocess MS
    mz_list = []
    intensity_list = []
    ms_mask_list = []
    for eims in EIMS:
        mz, intensity, ms_mask = pre_processing_ms(eims, config_json)
        mz_list.append(mz)
        intensity_list.append(intensity)
        ms_mask_list.append(ms_mask)

    mz = torch.cat(mz_list, dim=0).to(config_json["device"])
    intensity = torch.cat(intensity_list, dim=0).to(config_json["device"])
    ms_mask = torch.cat(ms_mask_list, dim=0).to(config_json["device"])
    
    return smiles_ids, smiles_mask, mz, intensity, ms_mask

def pre_processing_smi(smiles_str, tokenizer, config_json):
    if isinstance(smiles_str, str): 
        encode_dict = tokenizer(
            text=smiles_str, 
            max_length=config_json["dataset_config"]["smiles_maxlen"],
            padding='max_length', 
            truncation=True)
        smiles_ids = torch.from_numpy(
            np.array(encode_dict['input_ids'])).view(1, -1)  # smiles ----> tensor
        smiles_mask = torch.from_numpy(
            np.array(encode_dict['attention_mask'])).view(1, -1)
    else:
        smiles_ids = []
        smiles_mask = []
        for i in smiles_str:
            encode_dict = tokenizer(
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

def pre_processing_smi_with_canonical(smiles_str, tokenizer, config_json):
    if isinstance(smiles_str, str):
        smiles_str = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_str))
        encode_dict = tokenizer(
            text=smiles_str, 
            max_length=config_json["dataset_config"]["smiles_maxlen"],
            padding='max_length', 
            truncation=True)
        smiles_ids = torch.from_numpy(
            np.array(encode_dict['input_ids'])).view(1, -1)  # smiles ----> tensor
        smiles_mask = torch.from_numpy(
            np.array(encode_dict['attention_mask'])).view(1, -1)
    else:
        smiles_ids = []
        smiles_mask = []
        for i in smiles_str:
            encode_dict = tokenizer(
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

def pre_processing_ms(peaks_data, config_json):
    if isinstance(peaks_data, str):
        peaks_data = ast.literal_eval(peaks_data)
    num_peaks = config_json["dataset_config"]["ms_maxpeaks"]  # 100
    if len(peaks_data) <= num_peaks:
        ms_mask = torch.cat((torch.ones(len(peaks_data)), torch.zeros(num_peaks - len(peaks_data)))).view(1, -1) # 补零
        ms_tensor = torch.tensor(peaks_data).transpose(0, 1)
        mz, i = ms_tensor[0], ms_tensor[1]
        i = i / i.max()
        # mz 转为int类型
        mz = mz.int() # 转为int类型，不一样的地方
        ms_tensor = torch.stack([mz, i])
        ms_tensor = torch.cat((ms_tensor, torch.zeros(2, num_peaks - len(peaks_data))), dim=1)  # 补零
        mz, i = ms_tensor[0].view(1, -1), ms_tensor[1].view(1, -1)
    # 长度大于100，取i排行前100的峰
    elif len(peaks_data) > num_peaks:
        ms_mask = torch.ones(num_peaks).view(1, -1)
        peaks_data.sort(key=lambda x: x[1], reverse=True)
        # 取前100个峰再按m/z升序排序
        peaks_data = peaks_data[:num_peaks]
        peaks_data.sort(key=lambda x: x[0], reverse=False) # 按照m/z升序排列
        ms_tensor = torch.tensor(peaks_data).transpose(0, 1)
        mz, i = ms_tensor[0], ms_tensor[1]
        i = i / i.max()
        mz = mz.int() # 转为int类型，不一样的地方
        ms_tensor = torch.stack([mz, i]) # ms_tensor.shape = torch.Size([2, 100])
        mz, i = ms_tensor[0].view(1, -1), ms_tensor[1].view(1, -1)

    return [mz, i, ms_mask]


def prep_dataloader(nist, src_data_path, config_json, mode, shuffle=True, drop_last=False, num_workers=0, batch_size=None):
    if nist == 23:
        dataset = NIST23Dataset(src_data_path, config_json, mode=mode, data_end=config_json["dataset_config"]["data_num"])
    elif nist == 17:
        dataset = NIST17Dataset(src_data_path, config_json, mode=mode, data_end=config_json["dataset_config"]["data_num"])
    
    if batch_size is None:
        dataloader = DataLoader(dataset, config_json["training_config"]["batch_size"],
                                shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size,
                                shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers)
    return dataloader


def process_ms_spectra_for_wcs(ms, M_max = 1000):
    """
    将输入ms[[m/z, i],[m/z, i],...]转换为tensor[[m/z,..], [intensity,..]],再使用wcs加权得到ms_tensor
    """
    if isinstance(ms, str):
        ms = ast.literal_eval(ms)
    # 1. 将ms转换为tensor[[m/z,..], [intensity,..]]
    # M_max = 1000 最大的m/z值
    ms = torch.tensor(ms, dtype=torch.float32).T # 形状为[2, n] 
    # 2. 将m/z 和 iintensity按照WCS加权
    mz, intensity = ms[0], ms[1]
    intensity = intensity / intensity.max() # 归一化
    ms_ = mz * torch.sqrt(intensity) # WCS加权
    # 3.创建一个维度为M_max的tensor，初始化为0
    ms_tensor = torch.zeros(M_max).float()
    # 用加权后的值填充
    ms_indices = (mz-1).long()
    ms_tensor[ms_indices] = ms_
    # nan to 0
    ms_tensor[torch.isnan(ms_tensor)] = 0
    return ms_tensor


def get_tokenined_smiles(SMILES, config_path):
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    tokenizer=build_tokenizer(config_json['roberta_tokenizer_path'])
    smiles_data = pre_processing_smi(SMILES, tokenizer, config_json)[0].squeeze(0)
    smiles_token = []
    for i in smiles_data:
        if i == 2:
            break
        smiles_token.append(tokenizer.decode(i))
    smiles_token = smiles_token + ['</s>']
    return smiles_token
