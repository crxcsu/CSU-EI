import os
import json
import tqdm
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaModel
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup


def main(main_path, seed=2023, config_file="mlm_config.json", override_config=None):
    set_seed(seed)
    config_path = os.path.join(main_path, 'mlm_configs', config_file)
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())

    # --- Allow overriding config via arguments ---
    if override_config:
        print(f"\n[Info] Overriding configuration with: {override_config}")
        for key, value in override_config.items():
            if key in config_json and isinstance(config_json[key], dict) and isinstance(value, dict):
                config_json[key].update(value)
            else:
                config_json[key] = value

    if config_json["training_config"]["experiment_dir"] is None:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder_id = 'smiles_pretrain_nist{}_smil{}_bs{}_lr{:.1e}_{}'.format(
                    config_json["dataset_config"]["nist"], 
                    config_json["model_config"]["num_hidden_layers"],                
                    config_json["training_config"]["batch_size"],                   
                    config_json["training_config"]["lr"],  
                    time_stamp
                )
        config_json["training_config"]["experiment_dir"] = os.path.join("mlm_results", folder_id)
        LOG_DIR =  os.path.join(main_path, config_json["training_config"]["experiment_dir"])
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

    dataset_cfg = config_json["dataset_config"]
    training_cfg = config_json["training_config"]
    model_cfg = config_json["model_config"]

    train_dataloader = DataLoader(
        NISTMLMDataset(
            nist=dataset_cfg["nist"],
            mode='train',
            data_path=dataset_cfg["train_data_path"],
            data_end=dataset_cfg["data_end"],
            tokenizer = RobertaTokenizer.from_pretrained(config_json["roberta_tokenizer_path"]),
            mlm_probability=dataset_cfg["mlm_probability"],
            special_tokens_mask=dataset_cfg["special_tokens_mask"],
            prob_replace_mask=dataset_cfg["prob_replace_mask"],
            prob_replace_rand=dataset_cfg["prob_replace_rand"]
        ), 
        batch_size=training_cfg["batch_size"], 
        shuffle=True, 
        drop_last=False
    )
    
    dev_dataloader = DataLoader(
        NISTMLMDataset(
            nist=dataset_cfg["nist"],
            mode='dev',
            data_path=dataset_cfg["dev_data_path"],
            data_end=dataset_cfg["data_end"],
            tokenizer = RobertaTokenizer.from_pretrained(config_json["roberta_tokenizer_path"]),
            mlm_probability=dataset_cfg["mlm_probability"],
            special_tokens_mask=dataset_cfg["special_tokens_mask"],
            prob_replace_mask=dataset_cfg["prob_replace_mask"],
            prob_replace_rand=dataset_cfg["prob_replace_rand"]
        ), 
        batch_size=training_cfg["batch_size"], 
        shuffle=True, 
        drop_last=False
    )

    config = RobertaConfig(
        vocab_size=model_cfg["vocab_size"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        type_vocab_size=model_cfg["type_vocab_size"],
        hidden_dropout_prob=model_cfg["hidden_dropout_prob"],
        attention_probs_dropout_prob=model_cfg["attention_probs_dropout_prob"],
        hidden_size=model_cfg["hidden_size"]
    )
    roberta_mlm_model = RobertaForMaskedLM(config)
    if training_cfg['use_gradient_checkpoint']:
        roberta_mlm_model.gradient_checkpointing_enable()
        print("Gradient Checkpointing Enabled.")

    trainable_params = sum(p.numel() for p in roberta_mlm_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in roberta_mlm_model.parameters())
    print(f"Trainable params: {trainable_params}, Total params: {total_params}")

    train(train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, model=roberta_mlm_model, config_json=config_json, model_suffix=time_stamp, LOG_DIR=LOG_DIR)

    with open(os.path.join(main_path, 'mlm_configs', f"mlm_config_{time_stamp}.json"), 'w') as f:
        json.dump(config_json, f, indent=True)


class NISTMLMDataset(Dataset):
    def __init__(self,  nist,
                        mode, 
                        data_path, 
                        data_end,
                        tokenizer,
                        mlm_probability, 
                        special_tokens_mask, 
                        prob_replace_mask, 
                        prob_replace_rand):

        self.tokenizer = tokenizer
        if not os.path.exists(data_path):
            self.data = []
            if mode == 'train':
                for i in tqdm.tqdm(range(data_end)):
                    if nist == 23:
                        begin, end = i * 5000 + 1, (i + 1) * 5000
                        filename = "../data/nist23_data/train23/" + str(begin) + "-" + str(end) + ".json"
                    elif nist == 17:
                        begin, end = i * 1000 + 1, (i + 1) * 1000
                        filename = "../data/nist17_data/train_nist17/" + str(begin) + "-" + str(end) + ".json"
                    self._process_file(filename, mlm_probability, special_tokens_mask, prob_replace_mask, prob_replace_rand)
            elif mode == 'dev':
                if nist == 23:
                    filename = "../data/nist23_data/dev23/dev.json"
                elif nist == 17:
                    filename = "../data/nist17_data/dev_nist17/dev.json"
                self._process_file(filename, mlm_probability, special_tokens_mask, prob_replace_mask, prob_replace_rand)
            else:
                raise ValueError("mode should be 'train' or 'dev'")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            torch.save(self.data, data_path)
        else:
            print(f"Loading preprocessed {mode} dataset from {data_path} ...")
            self.data = torch.load(data_path)
            print(f"Successfully loaded {mode} dataset, total samples: {len(self.data)}")
    
    def _process_file(self, filename, mlm_probability, special_tokens_mask, prob_replace_mask, prob_replace_rand):
        with open(filename, 'r') as fp:
            file_data = json.load(fp)
        for tmp_data in file_data:
            input_ids, attention_mask = self.pre_processing_smi(tmp_data["smiles"])
            inputs, labels = self.mask_tokens(input_ids, mlm_probability, special_tokens_mask, prob_replace_mask, prob_replace_rand)   
            self.data.append((inputs, labels, attention_mask))
    
    def pre_processing_smi(self, smiles_str, smiles_maxlen=300):
        encode_dict = self.tokenizer.encode_plus(
            text=smiles_str, 
            max_length=smiles_maxlen,  
            padding='max_length',  
            truncation=True)
        smiles_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).view(1, -1)  # smiles ----> tensor
        smiles_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).view(1, -1)
        return [smiles_ids, smiles_mask]

    def mask_tokens(self, inputs, mlm_probability, special_tokens_mask, prob_replace_mask, prob_replace_rand):
        labels = inputs.clone()
        # Generate a matrix filled with 0.15, with the same dimensions as the inputs.
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # Here, "spcial_tokens" refers to "cls", "sep", "pad", etc. The dimension of "special_tokens_mask" is the same as that of "inputs".
        # The position of the special token is true, while the positions of the others are false.
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # The function of masked_fill_ is to set the values at the positions indicated by true in the special_tokens_mask to 0 in the probability_matrix.
        # The special_tokens_mask has already set the positions of special characters such as "cls" to true. That is, by setting the positions of special characters in the probability_matrix to 0, the special characters will not be masked.

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # Bert randomly masks 15% of the words in a sentence. That is, each word has a 15% chance of being masked (1) and an 85% chance of not being masked (0).
        # Therefore, it can be modeled as a Bernoulli distribution with parameter p = 0.15. Then, a mask matrix is generated using the Bernoulli function, with the same dimensions as the inputs.
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = prob_replace_rand / (1 - prob_replace_mask)
        indices_random = torch.bernoulli(torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape).int()
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    

def train(train_dataloader, dev_dataloader, model, config_json, model_suffix, LOG_DIR):
    config = config_json['training_config']
    device = config_json['device']
    early_stop_cnt = 0
    dev_min_loss = float('inf')
    save_model_path = LOG_DIR + f'/checkpoints/roberta_mlm_{model_suffix}.pth'
    loss_record_path = LOG_DIR + f'/loss_record/roberta_mlm_loss_record_{model_suffix}.csv'

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config['lr'], weight_decay=config['weight_decay'])

    if config["warm_up_step"]:
        total_steps = len(train_dataloader) * config['epochs']
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.15), #smiles_pretrain_nist23_smil6_bs128_lr7.5e-05_2026-01-23_162314预热0.08
                                                    num_training_steps=total_steps)

    scaler = GradScaler()
    model.to(device)
    for cur_epc in range(1, config['epochs']+1):
        model.train()
        print('\nmodel is mlm_training...  epoch:{}'.format(cur_epc))
        current_time = datetime.datetime.now()
        training_loss = 0
        for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc='Step')):
            optimizer.zero_grad()
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.squeeze(0).squeeze(1).to(device)
            labels = labels.squeeze(0).squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(0).squeeze(1).to(device)

            with autocast():
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.long()).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if config["warm_up_step"]:
                scheduler.step()
            scaler.update()
            # # ----------------不使用混合精度训练----------------
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            training_loss += loss.item()
        train_loss = training_loss / len(train_dataloader)

        dev_loss = dev(dev_dataloader, model, device)

        # Save model if your model improved
        if dev_loss < dev_min_loss:
            dev_min_loss = dev_loss
            torch.save(model.state_dict(), save_model_path)
            early_stop_cnt = 0
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(cur_epc, dev_min_loss))
        else:
            early_stop_cnt += 1
        
        current_data = {
                'timestamp': current_time,
                'epoch': cur_epc,
                'train_loss': train_loss,
                'dev_loss': dev_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'early_stop_cnt': early_stop_cnt
            }
        df_new = pd.DataFrame([current_data])
        df_new.to_csv(loss_record_path, mode='a', header=not os.path.exists(loss_record_path), index=False)
        # torch.save(roberta_mlm_model.roberta.embeddings.state_dict(), os.path.join(save_path, 'roberta_embed_mlm' + model_suffix + '.pth'))
        # torch.save(roberta_mlm_model.roberta.encoder.state_dict(), os.pmlm_ath.join(save_path, 'roberta_encoder_mlm' + model_suffix + '.pth'))

        torch.save(model.state_dict(), save_model_path)

        if cur_epc % 1 == 0:
            fig_save_path = LOG_DIR + f'/loss_record/plot_mlm_loss_record.png'
            plot_mlm_loss_curve(loss_record_path, fig_save_path)

        print("learning rate:{}".format(optimizer.param_groups[0]['lr']))
        print("epoch = {:4d}, Training loss: {}".format(cur_epc, train_loss))
        print("epoch = {:4d}, Dev loss: {}".format(cur_epc, dev_loss))
        if early_stop_cnt > config['early_stop']:
            break
    print('Finished training after {} epochs, dev_min_loss: {:.4f})'.format(cur_epc, dev_min_loss))
    
    convert_and_verify_model(save_model_path , config_json)


def dev(dev_dataloader, model, device):
    model.to(device)
    model.eval()
    dev_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(dev_dataloader, desc='Step')):
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.squeeze(0).squeeze(1).to(device)
            labels = labels.squeeze(0).squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(0).squeeze(1).to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.long()).loss
            dev_loss += loss.item()
    dev_loss /= len(dev_dataloader)
    return dev_loss


def convert_and_verify_model(model_path, config_json, Convert=True):
    """
    1. Load trained model weights
    2. Remove 'roberta.' prefix (adapt for RobertaModel)
    3. Remove 'lm_head.' related parameters
    4. Save as new _encoder.pth file
    5. Try loading into RobertaModel to verify parameter integrity
    """
    print(f"\n[Task: Convert & Verify Model] Processing: {os.path.basename(model_path)}")
    full_state_dict = torch.load(model_path, map_location='cpu')

    if Convert:
        # Convert keys: remove 'roberta.' prefix, discard 'lm_head.'
        encoder_state_dict = {k[8:]: v for k, v in full_state_dict.items() if k.startswith('roberta.')}
        # Save converted weights (Format: roberta_mlm_encoder_TIMESTAMP.pth)
        if 'roberta_mlm_' in os.path.basename(model_path):
            encoder_save_path = model_path.replace('roberta_mlm_', 'roberta_mlm_encoder_')
        else:
            encoder_save_path = model_path.replace('.pth', '_encoder.pth')
        torch.save(encoder_state_dict, encoder_save_path)
        print(f"Saved encoder weights to: {encoder_save_path}")

    try:
        # Quick verification
        model_cfg = config_json["model_config"]
        config = RobertaConfig(
            vocab_size=model_cfg["vocab_size"],
            max_position_embeddings=model_cfg["max_position_embeddings"],
            num_attention_heads=model_cfg["num_attention_heads"],
            num_hidden_layers=model_cfg["num_hidden_layers"],
            type_vocab_size=model_cfg["type_vocab_size"],
            hidden_dropout_prob=model_cfg["hidden_dropout_prob"],
            attention_probs_dropout_prob=model_cfg["attention_probs_dropout_prob"],
            hidden_size=model_cfg["hidden_size"]
        )
        verifier_model = RobertaModel(config, add_pooling_layer=False)
        verifier_model.load_state_dict(encoder_state_dict, strict=True)
        print("Verification passed (strict=True).")
    except Exception as e:
        print(f"Verification failed ({e}), retrying with strict=False...")
        missing_keys, unexpected_keys = verifier_model.load_state_dict(encoder_state_dict, strict=False)
        if len(missing_keys) > 0:
            print(f"Warning: {len(missing_keys)} missing keys in model")
        if len(unexpected_keys) > 0:
            print(f"Warning: {len(unexpected_keys)} unexpected keys in pretrained dict")


def setup_folders(LOG_DIR):
    # create the folder which will save experiment data.
    # CHECK_FOLDER = os.path.isdir(LOG_DIR)
    # if CHECK_FOLDER:
    #     shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR + '/checkpoints', exist_ok=True)
    # os.makedirs(LOG_DIR + '/training/data')
    # os.makedirs(LOG_DIR + '/training/plots')
    os.makedirs(LOG_DIR + '/loss_record', exist_ok=True)
    # os.makedirs(LOG_DIR + '/loss_record/plots', exist_ok=True)

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    print(f"Random seed set to {seed}")

def plot_mlm_loss_curve(loss_record_csv_path, fig_save_path):
    df = pd.read_csv(loss_record_csv_path)
    epoch_list = df['epoch'].tolist()
    train_loss_list = df['train_loss'].tolist()
    dev_loss_list = df['dev_loss'].tolist()
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    line1, = ax1.plot(epoch_list, dev_loss_list, color='blue', linewidth=1, label='dev_loss')
    line2, = ax1.plot(epoch_list, train_loss_list, color='red', linewidth=1, label='train_loss')
    ax1.tick_params(axis='y')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='right')
    fig.tight_layout()
    plt.savefig(fig_save_path, dpi=300)
    plt.close() 
 