import torch
import json
import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
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

# =============================================================================
# Section 1: Top-K Calculation Core Functions
# Originally from: fea_all_topk.py
# =============================================================================

#  --------------------Calculate Dot Product Matrix----------------
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
                       test_index_list_path='data/nist23_data/test23/test_index_list.json', 
                       mw_list_path='data/nist23_data/test23/uni_mainlib23_mw_list.json', 
                       formula_index_path='data/nist23_data/test23/uni_mainlib23_formula_index_dict.json', 
                       test_formula_list_path='data/nist23_data/test23/test_formula_list.json',
                       top_k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                       use_mass_filter=False, 
                       use_formula_filter=False,                        
                       MAIN_PATH=os.getcwd()):

    with open(os.path.join(MAIN_PATH, config_path), 'r') as f:
        config_json = json.loads(f.read())

    # 1 加载模型
    device = config_json['device']
    exp_dir = config_json["training_config"]["experiment_dir"]
    time_stamp = '_'.join(exp_dir.split('_')[-2:]) # 获取路径最后的时间戳部分
    model_path = os.path.join(MAIN_PATH, exp_dir, f'checkpoints/trained_model_{time_stamp}.pth')
    config_json["model_config"]["roberta_model_path"] = None
    model = CSUEI(**config_json["model_config"])
    model_dict = torch.load(model_path, map_location=device)
    # Handle DDP 'module.' prefix if present
    new_state_dict = {}
    for k, v in model_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # 2 计算mainlib all uni feature
    feature_path =  os.path.join(MAIN_PATH, exp_dir, f'mainlib_all_uni_feature/mainlib23_smiles_feature_{time_stamp}.pth')
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    if not os.path.exists(feature_path):
        mainlib_dataloader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                             config_json=config_json, mode='all', shuffle=False, batch_size=256)
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
    else:
        print("Feature file detected. Loading pre-computed features...")
        smiles_feature = torch.load(feature_path)

    # 3 计算test topk
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
    top_k = np.zeros(len(top_k_list))
    rank23 = []
    total_samples = len(full_test_index_list)

    for _, ms in tqdm(test_dataloader):
        ms[0], ms[1], ms[2] = ms[0].to(device), ms[1].to(device), ms[2].to(device)
        with torch.no_grad():
            zls,_ = model.ms_encoder(ms)
        ms_feature_tmp = zls / zls.norm(dim=-1, keepdim=True)  # 除以模长，归一化
        score = get_cos_distance(ms_feature_tmp, smiles_feature) # (B, 262157)
        B, D = score.shape
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
            print("use formula filter")
            test_formula_batch_index = full_test_formula_list_index[current_idx:current_idx+B]
            for i, index in enumerate(test_formula_batch_index):
                score[i, index] += 1
        
        topk, ranks = get_topk(score, test_index, top_k_list)
        rank23 += ranks
        topk_np = np.array(topk) * B / total_samples  # 乘以topk权重
        top_k += topk_np
        current_idx += B

    # test_pd = pd.read_csv('data/all_data/test23/test_smiles_rank23.csv')
    # test_pd['rank23'] = rank23
    # test_pd.to_csv('data/all_data/test23/test_smiles_rank23.csv', index=False)
    print(top_k)

def auto_eval_topk(config_path,
                   MAIN_PATH=os.getcwd(),
                   test_index_list_path='data/nist23_data/test23/test_index_list.json', 
                   mw_list_path='data/nist23_data/test23/uni_mainlib23_mw_list.json', 
                   formula_index_path='data/nist23_data/test23/uni_mainlib23_formula_index_dict.json', 
                   test_formula_list_path='data/nist23_data/test23/test_formula_list.json',
                   top_k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """
    Automated evaluation of Top-K accuracy across three filtering modes:
    1. No filter (Baseline)
    2. Mass filter (±5 Da)
    3. Formula filter
    
    Results are aggregated and saved to a CSV file.

    Args:
        config_path (str): Path to the experiment configuration JSON file (relative to MAIN_PATH).
        MAIN_PATH (str): Root directory of the project.
        test_index_list_path (str): Path to the JSON file containing ground truth indices for the test set.
        mw_list_path (str): Path to the JSON file containing molecular weights of all library molecules (for mass filtering).
        formula_index_path (str): Path to the JSON file mapping formulas to library indices (for formula filtering).
        test_formula_list_path (str): Path to the JSON file containing formulas for the test set molecules.
        top_k_list (list): List of 'k' values to calculate top-k accuracy for.
    """
    
    # -------------------- 1. Setup & Model Loading --------------------
    with open(os.path.join(MAIN_PATH, config_path), 'r') as f:
        config_json = json.loads(f.read())

    device = config_json['device']
    exp_dir = config_json["training_config"]["experiment_dir"]
    time_stamp = '_'.join(exp_dir.split('_')[-2:]) 
    
    # Initialize and load model
    model_path = os.path.join(MAIN_PATH, exp_dir, f'checkpoints/trained_model_{time_stamp}.pth')
    config_json["model_config"]["roberta_model_path"] = None
    model = CSUEI(**config_json["model_config"])
    
    # Load state dict handling DDP wrapper
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()

    # -------------------- 2. Precompute/Load Library Features --------------------
    feature_dir = os.path.join(MAIN_PATH, exp_dir, 'mainlib_all_uni_feature')
    feature_path = os.path.join(feature_dir, f'mainlib23_smiles_feature_{time_stamp}.pth')
    
    if os.path.exists(feature_path):
        print("Feature file detected. Loading pre-computed features...")
        smiles_feature = torch.load(feature_path)
        
    else:
        print("Feature file not found. Computing features...")
        os.makedirs(feature_dir, exist_ok=True)
        mainlib_loader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                         config_json=config_json, mode='all', shuffle=False, batch_size=256)
        
        smiles_features_list = []
        for smi, _ in tqdm(mainlib_loader, desc="Encoding Library SMILES"):
            smi = [x.to(device) for x in smi]
            with torch.no_grad():
                zis, _ = model.smiles_encoder(smi)
            smiles_features_list.append(zis / zis.norm(dim=-1, keepdim=True))
            
        smiles_feature = torch.cat(smiles_features_list, dim=0)
        torch.save(smiles_feature, feature_path)

    # -------------------- 3. Prepare Metadata for Filters --------------------
    # Load test indices
    with open(os.path.join(MAIN_PATH, test_index_list_path), 'r') as f:
        full_test_index_list = json.load(f)

    # Load and prepare MW data (for Mass Filter)
    with open(os.path.join(MAIN_PATH, mw_list_path), 'r') as f:
        mw_data_all = torch.tensor(json.load(f)).to(device)

    # Load and prepare Formula data (for Formula Filter)
    with open(os.path.join(MAIN_PATH, formula_index_path), 'r') as f:
        formula_index_dict = json.load(f)
    with open(os.path.join(MAIN_PATH, test_formula_list_path), 'r') as f:
        test_formula_list = json.load(f)
        
    # Pre-map test formulas to their library indices for fast lookup
    full_test_formula_indices = []
    for f_str in test_formula_list:
        indices = formula_index_dict.get(f_str, [])
        full_test_formula_indices.append(torch.tensor(indices, device=device))

    # -------------------- 4. Evaluation Loop --------------------
    test_loader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                  config_json=config_json, mode='test', shuffle=False, batch_size=256)
    
    modes = ['w/o mass filter', 'w/ mass filter ± 5Da', 'w/ formula filter']
    results = {mode: np.zeros(len(top_k_list)) for mode in modes}
    
    current_idx = 0
    total_samples = len(full_test_index_list)

    for _, ms in tqdm(test_loader, desc="Evaluating"):
        ms = [x.to(device) for x in ms] # [ms_bin, ms_inten, ms_mw]
        
        # Get MS embeddings
        with torch.no_grad():
            zls, _ = model.ms_encoder(ms)
        ms_feature_tmp = zls / zls.norm(dim=-1, keepdim=True)
        
        # Calculate Base Similarity Score (Cosine)
        base_score = get_cos_distance(ms_feature_tmp, smiles_feature)
        B, D = base_score.shape
        
        # Get ground truth indices for this batch
        batch_test_indices = torch.tensor(full_test_index_list[current_idx:current_idx+B]).to(device)

        # --- Mode 1: No Filter ---
        topk_1, _ = get_topk(base_score, batch_test_indices, top_k_list)
        results['w/o mass filter'] += np.array(topk_1) * B

        # --- Mode 2: Mass Filter (± 5Da) ---
        score_2 = base_score.clone()
        
        # Get MW for all library items (expanded to batch)
        mw_lib_expanded = mw_data_all.expand(B, D)
        # Get True MW for the query molecules
        mw_query = mw_data_all[batch_test_indices].unsqueeze(1).expand(B, D)
        
        # Apply mask
        mask = (mw_lib_expanded >= mw_query - 5) & (mw_lib_expanded <= mw_query + 5)
        score_2[~mask] = -1 # Mask out invalid candidates
        
        topk_2, _ = get_topk(score_2, batch_test_indices, top_k_list)
        results['w/ mass filter ± 5Da'] += np.array(topk_2) * B

        # --- Mode 3: Formula Filter ---
        score_3 = base_score.clone()
        
        batch_formula_indices = full_test_formula_indices[current_idx:current_idx+B]
        for i, valid_indices in enumerate(batch_formula_indices):
            if valid_indices.numel() > 0:
                score_3[i, valid_indices] += 1 # Boost valid formula candidates
                
        topk_3, _ = get_topk(score_3, batch_test_indices, top_k_list)
        results['w/ formula filter'] += np.array(topk_3) * B

        current_idx += B

    # -------------------- 5. Finalize & Save Results --------------------
    # Normalize accumulated counts by total samples to get average Top-K accuracy
    for key in results:
        results[key] /= total_samples

    df = pd.DataFrame.from_dict(results, orient='index', columns=top_k_list)
    print(df)
    
    # Save to CSV
    save_dir = os.path.join(MAIN_PATH, exp_dir, 'plot_result_analyzer')
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f'topk_evaluation_3modes_{time_stamp}.csv')
    df.to_csv(output_path)
    print(f"Results saved to {output_path}")


# =============================================================================
# Section 2: Visualization and Analysis Utilities
# Originally from: plot_eims.py
# =============================================================================
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



