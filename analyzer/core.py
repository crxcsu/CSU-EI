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
import ast

# =============================================================================
# Section 1: Top-K Calculation Core Functions
# Originally from: fea_all_topk.py
# =============================================================================

#  --------------------Calculate Dot Product Matrix----------------
def get_cos_distance(input_1, input_2):
    input_1 = input_1 / input_1.norm(dim=-1, keepdim=True)
    input_2 = input_2 / input_2.norm(dim=-1, keepdim=True)
    return input_1 @ input_2.t()

# -------------------------------------Calculate topk by batch---------------------------------
def get_topk(score, ms_index, k: list):
    """
    :param score(N, M): dot product matrix
    :param ms_index(N, 1): ms index, starts from 0, 1-dimensional, represents the database index of the true smiles corresponding to this mass spectrum
    :param k (int): topk
    """
    assert score.shape[0] == ms_index.shape[0]
   
    score_sorted = torch.argsort(score, dim=1, descending=True)  # return indices corresponding to sorted values
    tpk = []
    ranks = []
    for i in k:
        score_topk = score_sorted[:, :i]
        ms_matched = score_topk == ms_index.view(-1, 1)
        # Find indices of True entries in each row of ms_matched
        # ms_matched_index = torch.nonzero(ms_matched) # (N, 2)
        tpk.append((ms_matched.sum() / ms_index.shape[0]).item())

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

    # 1 Load the model
    device = config_json['device']
    exp_dir = config_json["training_config"]["experiment_dir"]
    time_stamp = '_'.join(exp_dir.split('_')[-2:]) 
    model_path = os.path.join(MAIN_PATH, exp_dir, f'checkpoints/trained_model_{time_stamp}.pth')

    config_json["model_config"]["roberta_model_path"] = None
    if "ms_use_se" not in config_json["model_config"]:
        config_json["model_config"]["ms_use_se"] = True

    model = CSUEI(**config_json["model_config"])
    # Handle DDP 'module.' prefix if present
    new_state_dict = {}
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # 2 Calculate mainlib all uni feature
    feature_path =  os.path.join(MAIN_PATH, exp_dir, f'mainlib_all_uni_feature/mainlib{config_json["dataset_config"]["nist"]}_smiles_feature_{time_stamp}.pth')
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    if not os.path.exists(feature_path):
        mainlib_dataloader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                             config_json=config_json, mode='all', shuffle=False, batch_size=256)
        smiles_list = []
        for smi, _ in tqdm(mainlib_dataloader):
                smi[0] = smi[0].to(device)
                smi[1] = smi[1].to(device)
                with torch.no_grad():
                    smi_feature, _ = model.smiles_encoder(smi)
                smiles_list.append(smi_feature)
        smiles_feature = torch.cat(smiles_list, dim=0)
        torch.save(smiles_feature, feature_path)
    else:
        print("Feature file detected. Loading pre-computed features...")
        smiles_feature = torch.load(feature_path)

    # 3 Calculate test topk
    test_dataloader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                     config_json=config_json, mode='test', shuffle=False, batch_size=256)
    
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
            ms_feature, _ = model.ms_encoder(ms)
        score = get_cos_distance(ms_feature, smiles_feature) # (B, 262157)
        B, D = score.shape
        test_index = torch.tensor(full_test_index_list[current_idx:current_idx+B]).to(device)
        
        # ---------------mw filter------------------
        if use_mass_filter:
            print("use mass filter")
            with open(os.path.join(MAIN_PATH, mw_list_path), 'r') as f:
                mw_list = json.load(f)
            mw_data = torch.tensor(mw_list).to(device)
            mw_data = mw_data.expand(B, D)
            mw_batch = mw_data.gather(1, test_index.unsqueeze(1)).expand(B, D) # (B, D)
            mw_low = mw_batch - 5
            mw_high = mw_batch + 5
            filter = (mw_data >= mw_low) & (mw_data <= mw_high) # Filter out those with a quality level of 5 or below.
            score[~filter] = -1
            
        # -----------------formula filter----------------
        if use_formula_filter:
            print("use formula filter")
            test_formula_batch_index = full_test_formula_list_index[current_idx:current_idx+B]
            for i, index in enumerate(test_formula_batch_index):
                score[i, index] += 1
        
        topk, ranks = get_topk(score, test_index, top_k_list)
        rank23 += ranks
        topk_np = np.array(topk) * B / total_samples  
        top_k += topk_np
        current_idx += B

    save_dir = os.path.join(MAIN_PATH, exp_dir, 'plot_result_analyzer')
    test_pd = pd.read_csv('data/nist23_data/test23/test_smiles_with_classification_20260504_233859.csv')
    test_pd['rank23'] = rank23
    test_pd = test_pd.drop_duplicates(subset=['smiles'], keep='first')    # Remove duplicate smiles
    test_pd.to_csv(os.path.join(save_dir, f'test_unismiles_with_classification_{time_stamp}.csv'), index=False)

    print(top_k)


def auto_eval_topk(config_path, dev_path=None,
                   MAIN_PATH=os.getcwd(),
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

    if dev_path is None:
        if config_json["dataset_config"]["nist"] == 23:
            test_index_list_path='data/nist23_data/test23/test_index_list.json'
            mw_list_path='data/nist23_data/test23/uni_mainlib23_mw_list.json'     
            formula_index_path='data/nist23_data/test23/uni_mainlib23_formula_index_dict.json'
            test_formula_list_path='data/nist23_data/test23/test_formula_list.json'   
        elif config_json["dataset_config"]["nist"] == 17:
            test_index_list_path='data/nist17_data/test_nist17/test_index_list.json'
            mw_list_path='data/nist17_data/test_nist17/uni_mainlib_mw_list.json'  
            formula_index_path='data/nist17_data/test_nist17/uni_mainlib17_formula_index_dict.json'
            test_formula_list_path='data/nist17_data/test_nist17/test_formula_list.json'
    else:
        test_index_list_path = os.path.join(MAIN_PATH, dev_path)
        if config_json["dataset_config"]["nist"] == 23:
            mw_list_path='data/nist23_data/test23/uni_mainlib23_mw_list.json'     
            formula_index_path='data/nist23_data/test23/uni_mainlib23_formula_index_dict.json'
            test_formula_list_path='data/nist23_data/test23/test_formula_list.json'   
        elif config_json["dataset_config"]["nist"] == 17:
            mw_list_path='data/nist17_data/test_nist17/uni_mainlib_mw_list.json'  
            formula_index_path='data/nist17_data/test_nist17/uni_mainlib17_formula_index_dict.json'
            test_formula_list_path='data/nist17_data/test_nist17/test_formula_list.json'

        
    model_path = os.path.join(MAIN_PATH, exp_dir, f'checkpoints/trained_model_{time_stamp}.pth')

    config_json["model_config"]["roberta_model_path"] = None
    if "ms_use_se" not in config_json["model_config"]:
        config_json["model_config"]["ms_use_se"] = True

    model = CSUEI(**config_json["model_config"])
    # Load state dict handling DDP wrapper
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()

    # -------------------- 2. Precompute/Load Library Features --------------------
    feature_dir = os.path.join(MAIN_PATH, exp_dir, 'mainlib_all_uni_feature')
    feature_path = os.path.join(feature_dir, f'mainlib{config_json["dataset_config"]["nist"]}_smiles_feature_{time_stamp}.pth')
    
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
                smi_feature, _ = model.smiles_encoder(smi)
            smiles_features_list.append(smi_feature)
            
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
    if dev_path is None:
        test_loader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                    config_json=config_json, mode='test', shuffle=False, batch_size=256)
    else:
        test_loader = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=MAIN_PATH, 
                                    config_json=config_json, mode='dev', shuffle=False, batch_size=256)
    
    modes = ['w/o filter', 'w/ mass filter ± 5Da', 'w/ formula filter']
    results = {mode: np.zeros(len(top_k_list)) for mode in modes}
    
    current_idx = 0
    total_samples = len(full_test_index_list)
    rank23 = []

    for _, ms in tqdm(test_loader, desc="Evaluating"):
        ms = [x.to(device) for x in ms] # [ms_bin, ms_inten, ms_mw]
        
        # Get MS embeddings
        with torch.no_grad():
            ms_feature, _ = model.ms_encoder(ms)
        
        # Calculate Base Similarity Score (Cosine)
        base_score = get_cos_distance(ms_feature, smiles_feature)
        B, D = base_score.shape
        
        # Get ground truth indices for this batch
        batch_test_indices = torch.tensor(full_test_index_list[current_idx:current_idx+B]).to(device)

        # --- Mode 1: No Filter ---
        topk_1, ranks = get_topk(base_score, batch_test_indices, top_k_list)
        rank23 += ranks
        results['w/o filter'] += np.array(topk_1) * B

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
    save_dir = os.path.join(MAIN_PATH, exp_dir, 'plot_result_analyzer')
    os.makedirs(save_dir, exist_ok=True)

    # Save Top-K accuracy table 
    for key in results:
        results[key] /= total_samples # Normalize Top-K results by total samples
    df = pd.DataFrame.from_dict(results, orient='index', columns=top_k_list)
    if dev_path is None:
        output_path = os.path.join(save_dir, f'topk_evaluation_3modes_{time_stamp}.csv')
    else:
        output_path = os.path.join(save_dir, f'dev_topk_evaluation_3modes_{time_stamp}.csv')
    df.to_csv(output_path)
    print(df)

    if dev_path is None and config_json["dataset_config"]["nist"] == 23:
        # Save per-molecule ranking results
        test_pd = pd.read_csv(os.path.join(MAIN_PATH, 'data/nist23_data/test23/test_smiles_with_classification_20260504_233859.csv'))
        test_pd['rank23'] = rank23  # Assign rank23 from evaluation
        test_pd = test_pd.drop_duplicates(subset=['smiles'], keep='first')  # Remove duplicate SMILES
        test_pd.to_csv(
            os.path.join(save_dir, f'test_unismiles_with_classification_{time_stamp}.csv'),
            index=False)


def get_topk_acc(config_path, k_list=['1', '5','10'], label='w/o filter', mode=None):
    exp_dir = os.path.dirname(config_path)
    time_stamp = '_'.join(exp_dir.split('_')[-2:])
    if mode is None: 
        topk_path = os.path.join(exp_dir, "plot_result_analyzer", f"topk_evaluation_3modes_{time_stamp}.csv")
    else:
        topk_path = os.path.join(exp_dir, "plot_result_analyzer", f"dev_topk_evaluation_3modes_{time_stamp}.csv")
    df = pd.read_csv(topk_path, index_col=0)
    values = df.loc[label, k_list].values * 100
    return list(values), df


def get_data_and_features(config_path, data, batch_size=1, epoch=None, MAIN_PATH=os.getcwd()):
    """
    Load trained model to extract features and compute similarity between SMILES and Mass Spectra.

    Args:
        config_path (str): Path to experiment config file.
        data (tuple): (SMILES_list, EIMS_list).
        batch_size (int): Batch size for inference.
        ms_maxpeaks (int, optional): Maximum number of peaks for MS data.
        MAIN_PATH (str): Project root directory.

    Returns:
        tuple: (normalized_smiles_features, normalized_ms_features, similarity_matrix), (smiles_ids, smiles_mask, mz, intensity, ms_mask)
    """
    with open(os.path.join(MAIN_PATH, config_path), 'r') as f:
            config_json = json.loads(f.read())
    config_json["model_config"]["roberta_model_path"] = None
    exp_dir = config_json["training_config"]["experiment_dir"]
    time_stamp = '_'.join(exp_dir.split('_')[-2:]) 

    smiles_ids, smiles_mask, mz, intensity, ms_mask = load_analysis_data(data, config_json)

    if epoch is not None and epoch != 0:
        model_path = os.path.join(MAIN_PATH, exp_dir, f'checkpoints/trained_model_epoch{epoch}_{time_stamp}.pth')
    elif epoch == 0:
        model_path = None
    else:    
        model_path = os.path.join(MAIN_PATH, exp_dir, f'checkpoints/trained_model_{time_stamp}.pth')

    # if "ms_use_se" not in config_json["model_config"]:
    #     config_json["model_config"]["ms_use_se"] = True
        
    model = CSUEI(**config_json["model_config"])
    if model_path is not None:
        model_dict = torch.load(model_path, map_location=config_json["device"])
        model.load_state_dict(model_dict, strict=False)
    model.to(config_json["device"])
    model.eval()

    test_smiles_feature = []
    test_ms_feature = []
    with torch.no_grad():
        for i in range(0, len(smiles_ids), batch_size):
            smiles_feature, smiles_score = model.smiles_encoder([smiles_ids[i:i+batch_size], smiles_mask[i:i+batch_size]])
            test_smiles_feature.append(smiles_feature)
        for i in range(0, len(mz), batch_size):
            ms_feature, ms_score = model.ms_encoder([mz[i:i+batch_size], intensity[i:i+batch_size], ms_mask[i:i+batch_size]])
            test_ms_feature.append(ms_feature)
    test_smiles_feature = torch.cat(test_smiles_feature, dim=0)
    test_ms_feature = torch.cat(test_ms_feature, dim=0)

    test_smiles_feature /= torch.norm(test_smiles_feature, dim=-1, keepdim=True)
    test_ms_feature /= torch.norm(test_ms_feature, dim=-1, keepdim=True)
    similarity = test_smiles_feature.cpu().numpy() @ test_ms_feature.cpu().numpy().T

    if len(smiles_ids)==1 and len(mz)==1:
        return (smiles_ids, smiles_mask, mz, intensity, ms_mask),(test_smiles_feature, test_ms_feature, similarity, smiles_score, ms_score)
    else:
        return (smiles_ids, smiles_mask, mz, intensity, ms_mask),(test_smiles_feature, test_ms_feature, similarity, None, None)


# =============================================================================
# Section 2: Visualization and Analysis Utilities
# Originally from: plot_eims.py
# =============================================================================
def plot_ms_spectrum(EIMS, save_dir=None, titles=None):
    i = 0
    for peaks_data in EIMS:
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
        
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        else:
            plt.title(f"Mass Spectrum {i+1}")
            
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"EIMS_{i+1}.png"), bbox_inches='tight')
        i += 1
        plt.show()


def plot_mol_structure(smiles_list, save_dir=None):
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem

    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    for i in range(len(mol_list)):
        mol = mol_list[i]
        if mol is None: 
            print(f"Warning: Invalid SMILES at index {i}")
            continue
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(300, 300))
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'SMILES {i+1}', fontsize=12)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"SMILES_{i+1}.png"), bbox_inches='tight')
        plt.show()


def plot_loss_curve(loss_record_csv_path, fig_save_path):
    df = pd.read_csv(loss_record_csv_path)
    epoch_list = df['epoch'].tolist()
    train_loss_list = df['train_loss'].tolist()
    dev_loss_list = df['dev_loss'].tolist()
    dev_top1_list = df['dev_top1'].tolist()
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    line1, = ax1.plot(epoch_list, dev_loss_list, color='blue', linewidth=1, label='dev_loss')
    line2, = ax1.plot(epoch_list, train_loss_list, color='red', linewidth=1, label='train_loss')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('top1')
    line3, = ax2.plot(epoch_list, dev_top1_list, color='green', linewidth=1, label='dev_top1')
    ax2.tick_params(axis='y')

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='right')
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=300)
    # plt.close()


def plot_distance_curve(distance_record_csv_path, fig_save_path):
    df = pd.read_csv(distance_record_csv_path)
    epoch_list = df['epoch'].tolist()
    train_distance_list = df['train_distance'].tolist()
    train_similarity_list = [1.0 - d for d in train_distance_list]
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Cosine Similarity', fontsize=20)
    line1, = ax1.plot(epoch_list, train_similarity_list, color='red', linestyle='--',
                      linewidth=1.2, label='Train Cosine Similarity')
    ax1.tick_params(axis='both', labelsize=20)
    lines = [line1]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='right', fontsize=20)
    fig.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format="svg",bbox_inches="tight", pad_inches=0.02)

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

# =============================================================================
# Section 3: Visualization and Analysis Utilities
# =============================================================================
def find_functional_group_pairs(json_path, smarts_transform, max_pairs=2):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    """
    Search for molecular pairs where one can be transformed into the other via the given reaction.
    This simulates finding pairs that differ by a specifc functional group.
    
    Args:
        json_path: Path to the dataset JSON.
        smarts_transform: SMARTS string defining the transformation (e.g., Remove OH).
        max_pairs: Stop after finding this many pairs.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Build Index: Canonical SMILES -> Data Entry (using walrus operator for conciseness)
    smi_to_data = {}
    for d in tqdm(data, desc="Indexing"):
        if (s := d.get('smiles')) and (m := Chem.MolFromSmiles(s)):
            smi_to_data[Chem.MolToSmiles(m, isomericSmiles=True)] = d

    rxn = AllChem.ReactionFromSmarts(smarts_transform)
    found_pairs = []

    # 2. Iterate and Transform
    for src_smi, src_data in tqdm(smi_to_data.items(), desc="Searching"):
        mol = Chem.MolFromSmiles(src_smi)
        # Run reaction, get all products
        for products in rxn.RunReactants((mol,)):
            for prod_mol in products:
                try:
                    Chem.SanitizeMol(prod_mol)
                    target_smi = Chem.MolToSmiles(prod_mol, isomericSmiles=True)
                except ValueError: continue

                # Match found: Product exists in our dataset
                if target_smi in smi_to_data and target_smi != src_smi:
                    found_pairs.append({
                        'with_group': src_data,
                        'without_group': smi_to_data[target_smi]
                    })
                    # Avoid duplicates from symmetric reaction sites by checking last added
                    if len(found_pairs) >= max_pairs: return found_pairs
    return found_pairs


def find_and_visualize_pairs(json_path, smarts_transform, max_pairs=5):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from IPython.display import display

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Build Index and Pre-filter
    # Indexing: Map Canonical SMILES to Data Entry
    smi_to_data = {}
    print("Indexing dataset...")
    for d in tqdm(data, desc="Indexing"):
        if (s := d.get('smiles')) and (m := Chem.MolFromSmiles(s)):
            try:
                smi_to_data[Chem.MolToSmiles(m, isomericSmiles=True)] = d
            except: continue
    
    # Create simplified pattern for filtering (remove map numbers for safer matching)
    # This prevents issues where map numbers in reaction SMARTS cause strict matching failures
    reactant_smarts = smarts_transform.split(">>")[0]
    # Remove map numbers (e.g., :1) for the substructure filter
    import re
    clean_pattern_smart = re.sub(r':\d+]', ']', reactant_smarts)
    reactant_pattern = Chem.MolFromSmarts(clean_pattern_smart)
    
    rxn = AllChem.ReactionFromSmarts(smarts_transform)
    found_count = 0

    print(f"Searching for pairs using transformation: {smarts_transform}")
    print(f"Filter pattern (stripped maps): {clean_pattern_smart}")

    # First pass: Identify all source candidates
    source_candidates = []
    print("Step 1: identifying candidate molecules containing the functional group...")
    for smi, d in tqdm(smi_to_data.items(), desc="Filtering Candidates"):
        mol = Chem.MolFromSmiles(smi)
        if reactant_pattern and mol.HasSubstructMatch(reactant_pattern):
            source_candidates.append((smi, d, mol))
    
    print(f"Found {len(source_candidates)} potential source molecules.")
    
    # 2. Iterate through candidates to find pairs
    print("Step 2: Generating products and searching for matches...")
    for src_smi, src_data, src_mol in tqdm(source_candidates, desc="Finding Pairs"):
        # Run reaction
        products_list = rxn.RunReactants((src_mol,))
        
        prod_smiles_set = set()
        for products in products_list:
            for prod_mol in products:
                try:
                    Chem.SanitizeMol(prod_mol)
                    prod_mol = Chem.RemoveHs(prod_mol) # Ensure canonical form matches database
                    prod_smi = Chem.MolToSmiles(prod_mol, isomericSmiles=True)
                    prod_smiles_set.add(prod_smi)
                except: continue
        
        # Check against dataset
        for target_smi in prod_smiles_set:
            if target_smi in smi_to_data and target_smi != src_smi:
                target_data = smi_to_data[target_smi]
                
                print(f"\n[MATCH FOUND] Pair {found_count + 1}")
                print(f"Source (ID {src_data.get('nist', 'N/A')}): {src_smi}")
                print(f"Target (ID {target_data.get('nist', 'N/A')}): {target_smi}")
                
                # Visualization (search_pair logic)
                img = Draw.MolsToGridImage(
                    [src_mol, Chem.MolFromSmiles(target_smi)], 
                    molsPerRow=2,
                    legends=[f"Source: {src_data.get('nist', 'N/A')}", f"Target: {target_data.get('nist', 'N/A')}"],
                    subImgSize=(300, 300)
                )
                display(img)
                
                found_count += 1
                if found_count >= max_pairs:
                    print(f"Reached max_pairs limit ({max_pairs}). Stopping.")
                    return


def search_pair(query_id, transform_smart, smi_map, id_map):
    from rdkit.Chem import AllChem, Draw
    from IPython.display import display
    q_data = id_map.get(str(query_id))
    if not q_data: return print(f"Query ID {query_id} not found.")
    
    q_mol = Chem.MolFromSmiles(q_data['smiles'])
    print(f"Query: ID {query_id} | SMILES: {q_data['smiles']}")

    try:
        rxn = AllChem.ReactionFromSmarts(transform_smart)
        products = rxn.RunReactants((q_mol,))
        
        found = False
        prod_smiles_set = set()
        
        for prod_group in products:
            for prod in prod_group:
                try:
                    Chem.SanitizeMol(prod)
                    prod_smi = Chem.MolToSmiles(prod, isomericSmiles=True)
                    prod_smiles_set.add(prod_smi)
                except: continue

        for prod_smi in prod_smiles_set:
            if target := smi_map.get(prod_smi):
                found = True
                print(f"\n[MATCH] Found Target ID: {target.get('nist')} | SMILES: {prod_smi}")
                # Set molsPerRow to match the number of molecules (2) to avoid empty whitespace
                img = Draw.MolsToGridImage([q_mol, Chem.MolFromSmiles(prod_smi)], 
                                         molsPerRow=2,
                                         legends=[f"query_ID: {query_id}", f"target_ID: {target.get('nist')}"],
                                         subImgSize=(300, 300))
                display(img)
                
        if not found: 
            print("No matching target found.")
            if prod_smiles_set:
                print(f"Generated SMILES: {list(prod_smiles_set)[:3]}")

    except Exception as e:
        print(f"Error: {e}")