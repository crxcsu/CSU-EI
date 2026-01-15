import os
import datetime
import json
import pandas as pd
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from exp.nt_xent_loss import NTXentLoss
# from model.CSU_EI import CSUEI
from model.CSU_EI import CSUEI
from data_provider.dataset import prep_dataloader
from utils.tools import setup_folders, set_seed
from analyzer.core import plot_loss_curve


def train_DDP(tr_set, dv_set, model, config_json, model_suffix, LOG_DIR):

    config = config_json['training_config']
    dist.init_process_group(backend='nccl') # 【集合通讯】其他进程连master，大家互认
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    # 断言：确保实际运行的 GPU 数量与配置文件一致
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
    loss = NTXentLoss(device, temperature=config['temperature'], alpha_weight=config['alpha_weight'])
    param_optimizer = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
       ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=config['lr'], weight_decay=config['weight_decay'])  # AdamW优化器,weight_decay是正则化项


    n_epochs = config['n_epochs']
    model_path = LOG_DIR + config['model_save_path'] + model_suffix + '.pth'
    model_temp_path = LOG_DIR + config['model_save_path_temp'] + model_suffix + '.pth'
    # loss_record_path = LOG_DIR + config['loss_record_path'] + model_suffix + '.pth'
    loss_record_csv_path = LOG_DIR + config['loss_record_path'] + model_suffix + '.csv'
    if os.path.exists(model_temp_path) and os.path.exists(loss_record_csv_path):
        state = torch.load(model_temp_path)
        if rank==0:
            model.load_state_dict(state['model'])
        # 读取CSV文件恢复训练状态
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
    
    # ------把预训练的Roberta模型的参数冻结----------------
    # for name, para in model.named_parameters():
    #     if "Smiles_model.model.embeddings" in name:
    #         para.requires_grad_(False)
    #     elif "Smiles_model.model.encoder" in name:
    #         para.requires_grad_(False)
    # -----------------冻结Roberta模型的参数-----------------
    # model.Smiles_model.freeze_backbone()

    # warm up 和 学习率衰减
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=config["warm_up_step"],
    #                                             num_training_steps=total_steps, num_cycles=20)  # restart 重新恢复学习率
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(config["warm_up_step"]/world_size),
                                                num_training_steps=len(train_loader)*n_epochs)

    scaler = GradScaler()   # 用于混合精度训练
    while epoch < n_epochs:
        train_sampler.set_epoch(epoch)
        model.train()
        epoch += 1
        if rank==0:
            print('\nmodel is training...  epoch:{}'.format(epoch))
        loss_train = 0
        count = 0
        current_time = datetime.datetime.now()
        # print("temperature:{}".format(loss.temperature)) # 打印出loss函数的temperature的值
        for smi, ms in tqdm(train_loader, disable=dist.get_rank() != 0):
            optimizer.zero_grad()
            smi[0] = smi[0].to(device)  # smi是个list，
            smi[1] = smi[1].to(device)  # smi[0]是input_ids, smi[1]是attention_mask
            ms[0] = ms[0].to(device)
            ms[1] = ms[1].to(device)
            ms[2] = ms[2].to(device)
            # ---------------- 混合精度训练----------------
            with autocast():
                smi_feature, smi_score, ms_feature, ms_score = model(smi, ms) 
                loss_epoch = loss(smi_feature, ms_feature)
            scaler.scale(loss_epoch).backward()
            scaler.step(optimizer)
            scheduler.step() 
            scaler.update()
            # # ----------------不使用混合精度训练----------------
            # smi_feature, smi_score, ms_feature, ms_score = model(smi, ms)
            # loss_epoch = loss(smi_feature, ms_feature)
            # loss_epoch.backward()
            # optimizer.step()
            # scheduler.step()
            if device == 'cpu':
                loss_train += loss_epoch.detach().cpu().item()
            else:
                loss_train += loss_epoch.detach().cuda().item()
            count += 1

        avg_loss_local = loss_train / count
        avg_loss_tensor = torch.tensor(avg_loss_local, device=device)
        dist.reduce(avg_loss_tensor, dst=0, op=dist.ReduceOp.SUM)

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
            current_data = {
                    'timestamp': current_time,
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'dev_loss': dev_loss,
                    'dev_top1': dev_top1,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'early_stop_cnt': early_stop_cnt
                }
            df_new = pd.DataFrame([current_data])
            df_new.to_csv(loss_record_csv_path, mode='a', header=not os.path.exists(loss_record_csv_path), index=False)

            # 每1个epoch保存一次loss曲线
            if epoch % 1 == 0:
                fig_save_path = LOG_DIR + f'/loss_record/plot_loss_record.png'
                plot_loss_curve(loss_record_csv_path, fig_save_path)

            # 打印学习率
            print("learning rate:{}".format(optimizer.param_groups[0]['lr']))
            print('epoch = {:4d}, train_loss = {:.4f}'.format(epoch, avg_train_loss))
            print('epoch = {:4d}, dev_loss = {:.4f}'.format(epoch, dev_loss), 'dev_top1_accuracy = {:.4f}'.format(dev_top1))

        # 等待 Rank 0 跑完验证
        dist.barrier() 

        # 广播 early_stop_cnt 给所有进程
        early_stop_tensor = torch.tensor(early_stop_cnt, dtype=torch.int, device=device)
        dist.broadcast(early_stop_tensor, src=0)
        early_stop_cnt = early_stop_tensor.item()
        if early_stop_cnt > config['early_stop']:
            break
        
    if rank==0:
        print('Finished training after {} epochs, max_dev_top1: {:.4f})'.format(epoch, max_dev_top1))


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
            smi_feature, smi_score, ms_feature, ms_score = model(smi, ms)   
            B = smi_feature.shape[0]
            i += B
            temp_loss = loss(smi_feature, ms_feature)
            smi_feature = smi_feature / smi_feature.norm(dim=-1, keepdim=True)  # 除以模长，归一化
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
    label = torch.arange(0, i).to(device)  # 对角线上为标签
    pred = torch.argmax(cos_distance, dim=1)  # 得到每一行余弦相似度最大的
    num_correct = (pred == label).sum().item()

    total_loss = total_loss / len(dv_set)
    top1_accuracy = num_correct / i
    return total_loss, top1_accuracy


def main(main_path, seed=2023, config_file="config.json"):

    set_seed(seed)
    config_path = os.path.join(main_path, 'configs', config_file)
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())

    if config_json["training_config"]["experiment_dir"] is None:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder_id = 'CSUEI_nist{}_smil{}_msl{}_bs{}_DDP{}_lr{:.1e}_smiesa{}_msesa{}_rm{}_{}'.format(
                    config_json["dataset_config"]["nist"],  # nist 数据集编号
                    config_json["model_config"]["num_hidden_layers"],  # smile_layer 缩写为 smil
                    config_json["model_config"]["ms_layer"],  # ms_layer 缩写为 msl                    
                    config_json["training_config"]["batch_size"],  # batch_size 缩写为 bs
                    config_json["training_config"]["use_distributed"],  # distributed data parallel 缩写为 DDP                    
                    config_json["training_config"]["lr"],  # learning rate 缩写为 lr
                    1 if config_json["model_config"]["smile_use_esa"] else 0,  # smile_use_esa 缩写为 smiesa
                    1 if config_json["model_config"]["ms_use_esa"] else 0,  # ms_use_esa 缩写为 msesa
                    1 if config_json["model_config"]["roberta_model_path"] else 0,  # roberta_model_path 缩写为 rm
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
    
    # 打印配置参数
    print("\n" + "="*30 + " Configuration " + "="*30)
    print(json.dumps(config_json, indent=4))
    print("="*75 + "\n")

    #  训练集和验证集
    tr_set = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=main_path, config_json=config_json, mode='train')
    dv_set = prep_dataloader(nist=config_json["dataset_config"]["nist"], src_data_path=main_path, config_json=config_json, mode='dev')

    # 使用**操作符将config_json["model_config"]字典解包为关键字参数，传递给ModelCLR类的构造函数，用于初始化模型
    model = CSUEI(**config_json["model_config"]) 
    # model.Smiles_model.freeze_backbone()

    # 计算模型可训练的参数量, 以及总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}, Total params: {total_params}")

    model_loss, model_loss_record = train_DDP(tr_set, dv_set, model, config_json, time_stamp, LOG_DIR)


def dev2(dv_set, model, device, local_rank):
    model.eval()
    
    loss_fn = NTXentLoss(device, temperature=0.1, alpha_weight=0.5)
    
    iterator = tqdm(dv_set) if local_rank == 0 else dv_set
    
    smi_list = []
    ms_list = []
    total_loss = 0
    
    with torch.no_grad():
        for smi, ms in iterator:
            ms[0], ms[1], ms[2] = ms[0].to(device), ms[1].to(device), ms[2].to(device)
            smi[0] = smi[0].to(device)
            smi[1] = smi[1].to(device)
            
            smi_feature, smi_score, ms_feature, ms_score = model(smi, ms)
            temp_loss = loss_fn(smi_feature, ms_feature)
            total_loss += temp_loss.item() * smi_feature.shape[0]
            
            smi_feature = smi_feature / smi_feature.norm(dim=-1, keepdim=True)
            ms_feature = ms_feature / ms_feature.norm(dim=-1, keepdim=True)
            
            smi_list.append(smi_feature)
            ms_list.append(ms_feature)
            
    local_smi_features = torch.cat(smi_list, dim=0)
    local_ms_features = torch.cat(ms_list, dim=0)
    local_count = local_smi_features.shape[0]
    
    world_size = dist.get_world_size()
    
    gathered_smi = [torch.zeros_like(local_smi_features) for _ in range(world_size)]
    gathered_ms = [torch.zeros_like(local_ms_features) for _ in range(world_size)]
    
    dist.all_gather(gathered_smi, local_smi_features)
    dist.all_gather(gathered_ms, local_ms_features)
    
    all_smi_features = torch.cat(gathered_smi, dim=0)
    all_ms_features = torch.cat(gathered_ms, dim=0)
    
    total_samples_tensor = torch.tensor([local_count], device=device)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    global_total_samples = total_samples_tensor.item()
    
    loss_tensor = torch.tensor([total_loss], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_avg_loss = loss_tensor.item() / global_total_samples
    
    cos_distance = all_ms_features @ all_smi_features.T
    label = torch.arange(0, all_ms_features.shape[0]).to(device)
    pred = torch.argmax(cos_distance, dim=1)
    num_correct = (pred == label).sum().item()
    top1_accuracy = num_correct / all_ms_features.shape[0]
    
    return global_avg_loss, top1_accuracy


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