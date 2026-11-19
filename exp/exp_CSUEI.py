import os
import datetime
import json
import pandas as pd
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from data_provider.dataset import prep_dataloader, DataLoader
from exp.nt_xent_loss import NTXentLoss
from model.CSU_EI import CSUEI
from utils.tools import setup_folders, set_seed
from analyzer.core import plot_loss_curve
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from analyzer.core import auto_eval_topk


def main(main_path, seed=2023, config_file="config.json"):
    set_seed(seed)
    config_path = os.path.join(main_path, 'configs', config_file)
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())

    if config_json["training_config"]["experiment_dir"] is None:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder_id = 'CSUEI_nist{}_smil{}_msl{}_bs{}_DDP{}_lr{:.1e}_smiesa{}_msesa{}_rm{}_se{}_{}'.format(
                    config_json["dataset_config"]["nist"],  # nist dataset index
                    config_json["model_config"]["num_hidden_layers"],  # smile_layer abbreviation: smil
                    config_json["model_config"]["ms_layer"],  # ms_layer abbreviation: msl                    
                    config_json["training_config"]["batch_size"],  # batch_size abbreviation: bs
                    config_json["training_config"]["use_distributed"],  # distributed data parallel abbreviation: DDP                    
                    config_json["training_config"]["lr"],  # learning rate abbreviation: lr
                    1 if config_json["model_config"]["smile_use_esa"] else 0,  # smile_use_esa abbreviation: smiesa
                    1 if config_json["model_config"]["ms_use_esa"] else 0,  # ms_use_esa abbreviation: msesa
                    1 if config_json["model_config"]["roberta_model_path"] else 0,  # roberta_model_path abbreviation: rm
                    1 if config_json["model_config"]["ms_use_se"] else 0,  # ms_use_se abbreviation: se
                    time_stamp
                )
        config_json["training_config"]["experiment_dir"] = os.path.join("results", folder_id)
        LOG_DIR =  os.path.join(main_path, config_json["training_config"]["experiment_dir"])
        setup_folders(LOG_DIR)
        with open(os.path.join(LOG_DIR, f"config_{time_stamp}.json"), 'w') as f:
            json.dump(config_json, f, indent=True)
    else:
        LOG_DIR = os.path.join(main_path, config_json["training_config"]["experiment_dir"])
        print('continue training in existing folder:', LOG_DIR)
        time_stamp = '_'.join(LOG_DIR.split('_')[-2:])
    
    # Print configuration parameters
    print("\n" + "="*30 + " Configuration " + "="*30)
    print(json.dumps(config_json, indent=4))
    print("="*75 + "\n")

    # Prepare datasets
    tr_set = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=main_path, config_json=config_json, mode='train')
    dv_set = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=main_path, config_json=config_json, mode='dev')

    # Unpack config_json["model_config"] dict as keyword arguments and pass to CSUEI constructor for model initialization
    model = CSUEI(**config_json["model_config"]) 
    # model.Smiles_model.freeze_backbone()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}, Total params: {total_params}")

    if config_json["training_config"]["use_distributed"]==1:
        train(tr_set, dv_set, model, config_json, time_stamp, LOG_DIR)
    else:
        train_DDP(tr_set, dv_set, model, config_json, time_stamp, LOG_DIR)
    
    auto_eval_topk(os.path.join(config_json["training_config"]["experiment_dir"], f"config_{time_stamp}.json"))
    auto_eval_topk(os.path.join(config_json["training_config"]["experiment_dir"], f"config_{time_stamp}.json"),
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
        state = torch.load(model_temp_path, map_location=device)
        model.load_state_dict(state)
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



