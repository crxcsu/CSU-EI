import torch
import json
import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_provider.dataset import *
from model.CSU_EI import CSUEI
from utils.tools import set_seed


# 对向量进行可视化
# from sklearn.manifold import TSNE
# import numpy as np
# import matplotlib.pyplot as plt

# tsne = TSNE(n_components=2, init='pca', random_state=0)
# smi_feature = smi_feature.cpu().numpy()
# ms_feature = ms_feature.cpu().numpy()
# feature = np.concatenate((smi_feature, ms_feature), axis=0)
# feature = tsne.fit_transform(feature)
# smi_feature = feature[:smi_feature.shape[0]]
# ms_feature = feature[smi_feature.shape[0]:]
# plt.figure()
# plt.scatter(smi_feature[:, 0], smi_feature[:, 1], c='r', label='smi')
# plt.scatter(ms_feature[:, 0], ms_feature[:, 1], c='b', label='ms')
# plt.legend()
# plt.savefig('data/feature/aspirin_feature/aspirin_feature_epoch{}.png'.format(epoch))


# def get_cos_distance(input_1, input_2):
#     # 归一化
#     input_1 = input_1 / input_1.norm(dim=-1, keepdim=True)
#     input_2 = input_2 / input_2.norm(dim=-1, keepdim=True)
#     with torch.no_grad():
#         return input_1 @ input_2.t()

# fea_all_topk.py
#  --------------------计算点积矩阵----------------
def get_cos_distance(input_1, input_2):
    with torch.no_grad():
        return input_1 @ input_2.t()
# -------------------------------------按batch计算topk---------------------------------
def get_topk(score, ms_index, k: list):
    """
    :param score(N, M): 点积矩阵
    :param ms_index(N, 1): ms索引, 从0开始, 一维, 表示此质谱对应真实smiles在数据库中的索引
    :param k (int): topk
    """
    assert score.shape[0] == ms_index.shape[0]
   
    score_sorted = torch.argsort(score, dim=1, descending=True) # 按排序结果返回对应值的索引
    tpk = []
    ranks = []
    for i in k:
        score_topk = score_sorted[:, :i]
        ms_matched = score_topk == ms_index.view(-1, 1)
        # 找到ms_matched中每一行为True的索引
        # ms_matched_index = torch.nonzero(ms_matched) # (N, 2)
        tpk.append((ms_matched.sum()/ms_index.shape[0]).item()) #/ms_index.shape[0]).item()

    for i in range(score.shape[0]):
        rank = (score_sorted[i] == ms_index[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    return tpk, ranks

def calculate_all_topk(config_path, 
                       test_index_list_path, 
                       mw_list_path, 
                       formula_index_path, 
                       test_formula_list_path,
                       top_k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                       use_mass_filter=False, 
                       use_formula_filter=False,                        
                       MAIN_PATH=os.getcwd()):
    """
    calculate_all_topk('models/config.json',
                    'data/all_data',
                    'data/model_save/trained_model_6+5_16epoc+512_2024-12-20_093126.pth',
                    'data/feature/mainlib23_smiles_feature-6+5_16epoc+512_2024-12-20_093126.pth',
                    'data/all_data/test23/test_index_list.json',
                    'data/all_data/test23/uni_mainlib23_mw_list.json', 
                    'data/all_data/test23/uni_mainlib23_formula_index_dict.json', 
                    'data/all_data/test23/test_formula_list.json',
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    use_mass_filter=False,
                    use_formula_filter=False)
    """
    set_seed(2023)
    with open(os.path.join(MAIN_PATH, config_path), 'r') as f:
        config_json = json.loads(f.read())
    # 1 加载模型
    device = config_json['device']
    exp_dir = config_json["training_config"]["experiment_dir"]
    time_stamp = '_'.join(exp_dir.split('_')[-2:]) # 获取路径最后的时间戳部分
    model_path = os.path.join(MAIN_PATH, exp_dir, f'checkpoints/trained_model_{time_stamp}.pth')
    model_dict = torch.load(model_path, map_location=device)
    model = CSUEI(**config_json["model_config"])
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()

    # 2 计算mainlib all uni feature
    feature_path =  os.path.join(MAIN_PATH, exp_dir, 'mainlib_all_uni_feature/mainlib23_smiles_feature.pth')
    if not os.path.exists(feature_path):
        mainlib_dataloader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                             config_json=config_json, mode='all', shuffle=False, batch_size=256)
        # mainlib_dataset = NIST17Dataset(data_path, config_json, mode='all', data_end=config_json["dataset_config"]["data_num"])
        # mainlib_dataloader = DataLoader(mainlib_dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False)
        smiles_list = []
        for smi, _ in tqdm(mainlib_dataloader):
                smi[0] = smi[0].to(device)
                smi[1] = smi[1].to(device)
                with torch.no_grad():
                    zis, _ = model.smiles_encoder(smi)
                smiles_feature_tmp = zis / zis.norm(dim=-1, keepdim=True)  # 除以模长，归一化
                smiles_list.append(smiles_feature_tmp)
        smiles_feature = torch.cat(smiles_list, dim=0)
        torch.save(smiles_feature, feature_path)
    smiles_feature = torch.load(feature_path)

    # 3 计算test topk
    top_k = np.zeros(len(top_k_list))
    rank23 = []
    test_dataloader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                     config_json=config_json, mode='test', shuffle=False, batch_size=256)
    
    # --- FIX: Load lists once before the loop ---
    with open(os.path.join(MAIN_PATH, test_index_list_path), 'r') as f:
        full_test_index_list = json.load(f)
    
    full_test_formula_list_index = None
    if use_formula_filter:
        with open(os.path.join(MAIN_PATH, formula_index_path), 'r') as f:
            formula_index_dict = json.load(f)
        with open(os.path.join(MAIN_PATH, test_formula_list_path), 'r') as f:
            test_formula_list = json.load(f)
        full_test_formula_list_index = [torch.tensor(formula_index_dict[i]).to(device) for i in test_formula_list if i in formula_index_dict]

    current_idx = 0
    total_samples = len(full_test_index_list)

    for _, ms in tqdm(test_dataloader):
        ms[0], ms[1], ms[2] = ms[0].to(device), ms[1].to(device), ms[2].to(device)
        with torch.no_grad():
            zls,_ = model.ms_encoder(ms)
        ms_feature_tmp = zls / zls.norm(dim=-1, keepdim=True)  # 除以模长，归一化
        score = get_cos_distance(ms_feature_tmp, smiles_feature) # (B, 262157)
        
        B, D = score.shape
        
        # --- FIX: Slice the correct indices for the current batch ---
        test_index = torch.tensor(full_test_index_list[current_idx:current_idx+B]).to(device)
        
        # ---------------mw filter------------------
        if use_mass_filter:
            print("use mass filter")
            with open(os.path.join(MAIN_PATH, mw_list_path), 'r') as f:
                mw_list = json.load(f)
            mw_data = torch.tensor(mw_list).to(device) # (262157)
            mw_data = mw_data.expand(B, D)
            mw_batch = mw_data.gather(1, test_index.unsqueeze(1)).expand(B, D) # (B, D)
            mw_low = mw_batch - 5
            mw_high = mw_batch + 5
            filter = (mw_data >= mw_low) & (mw_data <= mw_high) # 过滤出质量差在5以内的
            score[~filter] = -1
            
        # -----------------formula filter----------------
        if use_formula_filter:
            # --- FIX: Slice the correct formula indices ---
            test_formula_batch_index = full_test_formula_list_index[current_idx:current_idx+B]
            for i, index in enumerate(test_formula_batch_index):
                score[i, index] += 1

        #  计算一个batch的topk
        topk, ranks = get_topk(score, test_index, top_k_list)
        rank23 += ranks
        # --- FIX: Normalize with total sample count ---
        topk_np = np.array(topk) * B / total_samples  # 乘以topk权重
        top_k += topk_np
        
        current_idx += B

    # test_pd = pd.read_csv('data/all_data/test23/test_smiles_rank23.csv')
    # test_pd['rank23'] = rank23
    # test_pd.to_csv('data/all_data/test23/test_smiles_rank23.csv', index=False)
    print(top_k)


# plot_eims.py
def plot_ms_spectrum(peaks_data):
    if isinstance(peaks_data, str):
        peaks_data = ast.literal_eval(peaks_data)
    peaks = np.array(peaks_data, dtype=np.float32)
    mz = peaks[:, 0]
    mz_max = mz.max()
    intensity = peaks[:, 1]
    intensity = intensity / intensity.max()
    plt.figure(figsize=(10, 5))
    plt.bar(mz, intensity, width=mz_max/500, color='red')
    plt.xlabel("m/z")
    plt.ylabel("intensity")
    plt.show()


def plot_loss_curve(loss_record_csv_path, fig_save_path):
    # 绘制训练过程的loss和dev-top1曲线
    df = pd.read_csv(loss_record_csv_path)
    epoch_list = df['epoch'].tolist()
    train_loss_list = df['train_loss'].tolist()
    dev_loss_list = df['dev_loss'].tolist()
    dev_top1_list = df['dev_top1'].tolist()
    
    fig, ax1 = plt.subplots()
    # 绘制dev_loss曲线
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    line1, = ax1.plot(epoch_list, dev_loss_list, color='blue', linewidth=1, label='dev_loss')
    line2, = ax1.plot(epoch_list, train_loss_list, color='red', linewidth=1, label='train_loss')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('top1')
    line3, = ax2.plot(epoch_list, dev_top1_list, color='green', linewidth=1, label='dev_top1')
    ax2.tick_params(axis='y')

    # 创建一个图例包含所有的线
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='right')
    fig.tight_layout()
    # 保存高清图片
    plt.savefig(fig_save_path, dpi=300)
    # plt.close()

def convert_pth_to_csv(file_path):
    print(f"Loading data from: {file_path}")
    data = torch.load(file_path, map_location='cpu')
    
    # Handle tuple or dict format
    loss_record = data[0] if isinstance(data, tuple) else data
    
    # Create DataFrame directly from zipped lists
    train_data = loss_record['train']
    dev_data = loss_record['dev']
    min_len = min(len(train_data), len(dev_data))
    
    # Extract columns
    epochs = [t[0] for t in train_data[:min_len]]
    train_losses = [t[1] for t in train_data[:min_len]]
    timestamps = [t[2] for t in train_data[:min_len]]
    dev_losses = [d[1] for d in dev_data[:min_len]]
    dev_top1s = [d[2] if len(d) > 2 else None for d in dev_data[:min_len]]
    
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'dev_loss': dev_losses,
        'dev_top1': dev_top1s,
        'timestamp': timestamps,
        'early_stop_cnt': [0] * (min_len - 1) + [loss_record.get('early_stop_cnt', 0)]
    })

    # Save to CSV
    csv_path = file_path.replace('.pth', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to: {csv_path}")
    
    # Print minimum dev loss
    if 'dev_loss' in df.columns:
        min_row = df.loc[df['dev_loss'].idxmin()]
        print(f"Minimum Dev Loss: {min_row['dev_loss']:.6f} at epoch {int(min_row['epoch'])}")
    
    print("Head of DataFrame:")
    print(df.head())
    
    return df


def save_aspirin_feature(model,config_json, epoch):
    model.eval()
    smi_feature_list = []
    ms_feature_list = []
    with open('ycx_7_mstest/smiles_peak_data.csv', 'r') as f:
        reader = csv.reader(f)

    for i, row in tqdm(enumerate(reader)):
        if i == 0:
            continue
        smiles = row[0]
        peak_data = row[1]

        tokenizer = build_tokenizer(config_json['roberta_tokenizer_path'])
        ms = pre_processing_ms(peak_data, config_json)
        smi = pre_processing_smi(smiles, tokenizer, config_json)
        smi[0] = smi[0].to(config_json['device'])
        smi[1] = smi[1].to(config_json['device'])
        ms[0] = ms[0].to(config_json['device'])
        ms[1] = ms[1].to(config_json['device'])
        ms[2] = ms[2].to(config_json['device'])
        with torch.no_grad():
            smi_feature, smi_score, ms_feature, ms_score = model(smi, ms)
            smi_feature_list.append(smi_feature)
            ms_feature_list.append(ms_feature)
    smi_feature = torch.cat(smi_feature_list, dim=0)
    ms_feature = torch.cat(ms_feature_list, dim=0)
            
    torch.save([smi_feature, ms_feature], 'data/feature/aspirin_feature/aspirin_feature_epoch{}.pth'.format(epoch))



