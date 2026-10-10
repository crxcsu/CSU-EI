import os
import datetime
import json
import numpy as np  
import pandas as pd
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from data_provider.dataset import prep_dataloader, DataLoader
from exp.nt_xent_loss import NTXentLoss
from model.ModelCLR import build_model_from_config
from utils.tools import setup_folders, set_seed
from analyzer.core import plot_loss_curve
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from analyzer.core import get_cos_distance, get_topk   

def auto_eval_topk(config_path, model, dev_path=None,
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


def main(main_path, seed=2023, config_file="config_ModelCLR.json"):
    set_seed(seed)
    config_path = os.path.join(main_path, 'configs', config_file)
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())

    if config_json["training_config"]["experiment_dir"] is None:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        cfg_m = config_json["model_config"]
        folder_id = f'baseline_{time_stamp}'
           
        config_json["training_config"]["experiment_dir"] = os.path.join("results", folder_id)
        LOG_DIR = os.path.join(main_path, config_json["training_config"]["experiment_dir"])
        setup_folders(LOG_DIR)
        with open(os.path.join(LOG_DIR, f"config_{time_stamp}.json"), 'w') as f:
            json.dump(config_json, f, indent=True)
    else:
        LOG_DIR = os.path.join(main_path, config_json["training_config"]["experiment_dir"])
        print('continue training in existing folder:', LOG_DIR)
        time_stamp = '_'.join(LOG_DIR.split('_')[-2:])

    print("\n" + "="*30 + " Configuration " + "="*30)
    print(json.dumps(config_json, indent=4))
    print("="*75 + "\n")

    tr_set = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=main_path, config_json=config_json, mode='train')
    dv_set = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=main_path, config_json=config_json, mode='dev')

    # build model via factory function
    model = build_model_from_config(config_json["model_config"])
    # model.Smiles_model.freeze_backbone()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}, Total params: {total_params}")

    if config_json["training_config"]["use_distributed"]==1:
        train(tr_set, dv_set, model, config_json, time_stamp, LOG_DIR)
    else:
        train_DDP(tr_set, dv_set, model, config_json, time_stamp, LOG_DIR)
    
    auto_eval_topk(os.path.join(config_json["training_config"]["experiment_dir"], f"config_{time_stamp}.json"),
                   model)
    auto_eval_topk(os.path.join(config_json["training_config"]["experiment_dir"], f"config_{time_stamp}.json"),
                   model,
                   dev_path="data/nist23_data/dev23/dev_index_list.json")
    
    with open(os.path.join(main_path, 'configs', f"config_{time_stamp}.json"), 'w') as f:
        json.dump(config_json, f, indent=True)


def train(tr_set, dv_set, model, config_json, model_suffix, LOG_DIR):
    config = config_json['training_config']
    device = config_json['device']
    n_epochs = config['n_epochs']
    
    model_path = LOG_DIR + config['model_save_path'] + model_suffix + '.pth'
    model_temp_path = LOG_DIR + config['model_save_path_temp'] + model_suffix + '.pth'
    # loss_record_path = LOG_DIR + config['loss_record_path'] + model_suffix + '.pth'
    loss_record_csv_path = LOG_DIR + config['loss_record_path'] + model_suffix + '.csv'

    loss = NTXentLoss(device, temperature=config['temperature'], alpha_weight=config['alpha_weight'])
    param_optimizer = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
       ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=config['lr'], weight_decay=config['weight_decay'])

    if os.path.exists(model_temp_path) and os.path.exists(loss_record_csv_path):
        state = torch.load(model_temp_path)
        model.load_state_dict(state['model'], strict=False)
        df = pd.read_csv(loss_record_csv_path)
        early_stop_cnt = int(df.iloc[-1]['early_stop_cnt'])
        epoch = int(df.iloc[-1]['epoch'])
        max_dev_top1 = df['dev_top1'].max()
        dev_min_loss = df.loc[df['dev_top1'].idxmax()]['dev_loss']
    else:
        early_stop_cnt = 0
        epoch = 0
        max_dev_top1 = 0
        dev_min_loss = float('inf')
    
    # ------Freeze parameters of pretrained Roberta model----------------
    # for name, para in model.named_parameters():
    #     if "Smiles_model.model.embeddings" in name:
    #         para.requires_grad_(False)
    #     elif "Smiles_model.model.encoder" in name:
    #         para.requires_grad_(False)
    # -----------------Freeze Roberta model parameters-----------------
    # model.Smiles_model.freeze_backbone()

    # Warm up and learning rate decay
    total_steps = len(tr_set) * n_epochs
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=config["warm_up_step"],
    #                                             num_training_steps=total_steps, num_cycles=20)  # restart recover learning rate
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config["warm_up_step"],
                                                num_training_steps=total_steps)

    scaler = GradScaler()   # For mixed precision training
    model.to(device)
    
    while epoch < n_epochs:
        model.train()
        epoch += 1
        print('\nmodel is training...  epoch:{}'.format(epoch))
        current_time = datetime.datetime.now()
        loss_train = 0
        distance_train = 0
        count = 0
        for smi, ms in tqdm(tr_set):
            optimizer.zero_grad()
            smi[0] = smi[0].to(device) 
            smi[1] = smi[1].to(device)  
            ms[0] = ms[0].to(device)
            ms[1] = ms[1].to(device)
            ms[2] = ms[2].to(device)
            # ---------------- Mixed precision training----------------
            with autocast():
                # smi_smi, smi_mask = smi
                # ms_mz, ms_i, ms_mask = ms
                # smi_feature, smi_score, ms_feature, ms_score = model(smi_smi, smi_mask, ms_mz, ms_i, ms_mask) 
                smi_feature, smi_score, ms_feature, ms_score = model(smi, ms) 
                loss_epoch = loss(smi_feature, ms_feature)
                smi_norm = torch.nn.functional.normalize(smi_feature, dim=-1)
                ms_norm = torch.nn.functional.normalize(ms_feature, dim=-1)
                distance_epoch = (1 - torch.nn.functional.cosine_similarity(smi_norm, ms_norm, dim=-1)).mean()
            scaler.scale(loss_epoch).backward()
            scaler.step(optimizer)
            scheduler.step() 
            scaler.update()
            # # ----------------Training without mixed precision----------------
            # smi_feature, smi_score, ms_feature, ms_score = model(smi, ms)
            # loss_epoch = loss(smi_feature, ms_feature)
            # loss_epoch.backward()
            # optimizer.step()
            # scheduler.step()

            if device == 'cpu':
                loss_train += loss_epoch.detach().cpu().item()
                distance_train += distance_epoch.detach().cpu().item()
            else:
                loss_train += loss_epoch.detach().cuda().item()
                distance_train += distance_epoch.detach().cuda().item()
            count += 1
        
        # After each epoch, test your model on the validation (development) set.
        dev_loss, dev_top1 = dev(dv_set, model, device)

        if epoch in (1, 3):
            torch.save(
                model.state_dict(),
                LOG_DIR + config['model_save_path'] + f"epoch{epoch}_" + model_suffix + '.pth'
            )
            print(f'Saving model at epoch = {epoch:4d}')

        if dev_top1 > max_dev_top1:
            # Save model if your model improved
            dev_min_loss = dev_loss
            max_dev_top1 = dev_top1
            torch.save(model.state_dict(), model_path)
            early_stop_cnt = 0
            print('Saving best model (epoch = {:4d}, loss = {:.4f})'.format(epoch, dev_min_loss))
        else:
            early_stop_cnt += 1
        
        torch.save(model.state_dict(), model_temp_path)

        current_data = {
                'timestamp': current_time,
                'epoch': epoch,
                'train_loss': loss_train / count,
                'train_distance': distance_train / count,
                'dev_loss': dev_loss,
                'dev_top1': dev_top1,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'early_stop_cnt': early_stop_cnt
            }
        df_new = pd.DataFrame([current_data])
        df_new.to_csv(loss_record_csv_path, mode='a', header=not os.path.exists(loss_record_csv_path), index=False)

        if epoch % 1 == 0:
            fig_save_path = LOG_DIR + f'/loss_record/plot_loss_record.png'
            plot_loss_curve(loss_record_csv_path, fig_save_path)

        print("learning rate:{}".format(optimizer.param_groups[0]['lr']))
        print('epoch = {:4d}, train_loss = {:.4f}'.format(epoch, loss_train / count))
        print('epoch = {:4d}, dev_loss = {:.4f}'.format(epoch, dev_loss), 'dev_top1_accuracy = {:.4f}'.format(dev_top1))
        if early_stop_cnt > config['early_stop']:
            break
    
    print('Finished training after {} epochs, max_dev_top1: {:.4f})'.format(epoch, max_dev_top1))
    
    if os.path.exists(model_temp_path):
        os.remove(model_temp_path)
        print(f"Removed temporary model file: {model_temp_path}")


def train_DDP(tr_set, dv_set, model, config_json, model_suffix, LOG_DIR):
    config = config_json['training_config']
    dist.init_process_group(backend='nccl')
    rank=dist.get_rank()
    world_size=dist.get_world_size()

    assert world_size == config['use_distributed'], \
        f"Configured for {config['use_distributed']} GPUs but running with {world_size} GPUs."
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Re-create DataLoader with DistributedSampler
    train_dataset = tr_set.dataset
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
                                train_dataset, 
                                batch_size=tr_set.batch_size, 
                                sampler=train_sampler, 
                                num_workers=tr_set.num_workers,
                                drop_last=tr_set.drop_last)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss = NTXentLoss(device, temperature=config['temperature'], alpha_weight=config['alpha_weight'], use_distributed=config['use_distributed'])
    param_optimizer = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
       ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    n_epochs = config['n_epochs']
    model_path = LOG_DIR + config['model_save_path'] + model_suffix + '.pth'
    model_temp_path = LOG_DIR + config['model_save_path_temp'] + model_suffix + '.pth'
    # loss_record_path = LOG_DIR + config['loss_record_path'] + model_suffix + '.pth'
    loss_record_csv_path = LOG_DIR + config['loss_record_path'] + model_suffix + '.csv'
    if os.path.exists(model_temp_path) and os.path.exists(loss_record_csv_path):
        state = torch.load(model_temp_path)
        if rank==0:
            model.load_state_dict(state['model'], strict=False)
        df = pd.read_csv(loss_record_csv_path)
        early_stop_cnt = int(df.iloc[-1]['early_stop_cnt'])
        epoch = int(df.iloc[-1]['epoch'])
        max_dev_top1 = df['dev_top1'].max()
        dev_min_loss = df.loc[df['dev_top1'].idxmax()]['dev_loss']
    else:
        early_stop_cnt = 0
        epoch = 0
        max_dev_top1 = 0
        dev_min_loss = float('inf')
    
    # ------Freeze parameters of pretrained Roberta model----------------
    # for name, para in model.named_parameters():
    #     if "Smiles_model.model.embeddings" in name:
    #         para.requires_grad_(False)
    #     elif "Smiles_model.model.encoder" in name:
    #         para.requires_grad_(False)
    # -----------------Freeze Roberta model parameters-----------------
    # model.Smiles_model.freeze_backbone()

    # Warm up and learning rate decay
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=config["warm_up_step"],
    #                                             num_training_steps=total_steps, num_cycles=20)  # restart recover learning rate
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config["warm_up_step"],
                                                num_training_steps=len(train_loader)*n_epochs)

    scaler = GradScaler()   # For mixed precision training
    while epoch < n_epochs:
        train_sampler.set_epoch(epoch)
        model.train()
        epoch += 1
        if rank==0:
            print('\nmodel is training...  epoch:{}'.format(epoch))
        loss_train = 0
        distance_train = 0
        count = 0
        current_time = datetime.datetime.now()
        # print("temperature:{}".format(loss.temperature)) # Print temperature value of loss function
        for smi, ms in tqdm(train_loader, disable=dist.get_rank() != 0):
            optimizer.zero_grad()
            smi[0] = smi[0].to(device)  # smi is a list
            smi[1] = smi[1].to(device)  # smi[0] is input_ids, smi[1] is attention_mask
            ms[0] = ms[0].to(device)
            ms[1] = ms[1].to(device)
            ms[2] = ms[2].to(device)
            # ---------------- Mixed precision training----------------
            with autocast():
                smi_feature, smi_score, ms_feature, ms_score = model(smi, ms) 
                loss_epoch = loss(smi_feature, ms_feature)
                smi_norm = torch.nn.functional.normalize(smi_feature, dim=-1)
                ms_norm = torch.nn.functional.normalize(ms_feature, dim=-1)
                distance_epoch = (1 - torch.nn.functional.cosine_similarity(smi_norm, ms_norm, dim=-1)).mean()
            scaler.scale(loss_epoch).backward()
            scaler.step(optimizer)
            scheduler.step() 
            scaler.update()
            # # ----------------Training without mixed precision----------------
            # smi_feature, smi_score, ms_feature, ms_score = model(smi, ms)
            # loss_epoch = loss(smi_feature, ms_feature)
            # loss_epoch.backward()
            # optimizer.step()
            # scheduler.step()
            if device == 'cpu':
                loss_train += loss_epoch.detach().cpu().item()
                distance_train += distance_epoch.detach().cpu().item()
            else:
                loss_train += loss_epoch.detach().cuda().item()
                distance_train += distance_epoch.detach().cuda().item()
            count += 1

        avg_loss_local = loss_train / count
        avg_loss_tensor = torch.tensor(avg_loss_local, device=device)
        dist.reduce(avg_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        avg_distance_local = distance_train / count
        avg_distance_tensor = torch.tensor(avg_distance_local, device=device)
        dist.reduce(avg_distance_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank==0:
            # After each epoch, test your model on the validation (development) set.
            raw_model=model.module
            dev_loss, dev_top1 = dev(dv_set, raw_model, device)

            # early stopping
            if dev_top1 > max_dev_top1:
                # Save model if your model improved
                dev_min_loss = dev_loss
                max_dev_top1 = dev_top1
                torch.save(model.state_dict(), model_path)
                early_stop_cnt = 0
                print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch, dev_min_loss))
            else:
                early_stop_cnt += 1            

            # record
            avg_train_loss = avg_loss_tensor.item() / world_size
            avg_train_distance = avg_distance_tensor.item() / world_size
            current_data = {
                    'timestamp': current_time,
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'train_distance': avg_train_distance,
                    'dev_loss': dev_loss,
                    'dev_top1': dev_top1,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'early_stop_cnt': early_stop_cnt
                }
            df_new = pd.DataFrame([current_data])
            df_new.to_csv(loss_record_csv_path, mode='a', header=not os.path.exists(loss_record_csv_path), index=False)

            # Save loss curve every 1 epoch
            if epoch % 1 == 0:
                fig_save_path = LOG_DIR + f'/loss_record/plot_loss_record.png'
                plot_loss_curve(loss_record_csv_path, fig_save_path)

            # Print learning rate
            print("learning rate:{}".format(optimizer.param_groups[0]['lr']))
            print('epoch = {:4d}, train_loss = {:.4f}'.format(epoch, avg_train_loss))
            print('epoch = {:4d}, dev_loss = {:.4f}'.format(epoch, dev_loss), 'dev_top1_accuracy = {:.4f}'.format(dev_top1))

        # Wait for Rank 0 to finish validation
        dist.barrier() 

        # Broadcast early_stop_cnt to all processes
        early_stop_tensor = torch.tensor(early_stop_cnt, dtype=torch.int, device=device)
        dist.broadcast(early_stop_tensor, src=0)
        early_stop_cnt = early_stop_tensor.item()
        if early_stop_cnt > config['early_stop']:
            break
        
    if rank==0:
        print('Finished training after {} epochs, max_dev_top1: {:.4f})'.format(epoch, max_dev_top1))

        # Delete temporary model file
        if os.path.exists(model_temp_path):
            os.remove(model_temp_path)
            print(f"Removed temporary model file: {model_temp_path}")


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    i = 0
    smi_list = []
    ms_list = []
    # loss_smi = nn.CrossEntropyLoss().to(device)
    # loss_ms = nn.CrossEntropyLoss().to(device)
    loss = NTXentLoss(device, temperature=0.1, alpha_weight=0.5)
    for smi, ms in tqdm(dv_set):
        ms[0], ms[1], ms[2] = ms[0].to(device), ms[1].to(device), ms[2].to(device)
        # smi = smi.to(device)
        smi[0] = smi[0].to(device)
        smi[1] = smi[1].to(device)
        with torch.no_grad():
            # smi_smi, smi_mask = smi
            # ms_mz, ms_i, ms_mask = ms
            # smi_feature, smi_score, ms_feature, ms_score = model(smi_smi, smi_mask, ms_mz, ms_i, ms_mask)
            smi_feature, smi_score, ms_feature, ms_score = model(smi, ms)
            B = smi_feature.shape[0]
            i += B
            temp_loss = loss(smi_feature, ms_feature)
            smi_feature = smi_feature / smi_feature.norm(dim=-1, keepdim=True)  # Normalize by norm
            ms_feature = ms_feature / ms_feature.norm(dim=-1, keepdim=True)
            smi_list.append(smi_feature)
            ms_list.append(ms_feature)
        if device == 'cpu':
            total_loss += temp_loss.detach().cpu().item()
        else:
            total_loss += temp_loss.detach().cuda().item()

    smi_feature = torch.cat(smi_list, dim=0)
    ms_feature = torch.cat(ms_list, dim=0)
    cos_distance = ms_feature @ smi_feature.T
    label = torch.arange(0, i).to(device)  # Labels lie on the diagonal
    pred = torch.argmax(cos_distance, dim=1)  # Get indices with maximum cosine similarity for each row
    num_correct = (pred == label).sum().item()

    total_loss = total_loss / len(dv_set)
    top1_accuracy = num_correct / i
    return total_loss, top1_accuracy



