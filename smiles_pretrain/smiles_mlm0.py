import os
import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup



data_end = 70
batch_size = 128
epochs = 20
learning_rate = 5e-5
weight_decay = 0.0001
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
dev_data_path = 'data/all_data/mlm_data/dev23_mlm.pth'
train_data_path = 'data/all_data/mlm_data/train23_mlm.pth'


mlm_probability=0.15
special_tokens_mask=None
prob_replace_mask=0.8
prob_replace_rand=0.1
prob_keep_ori=0.1

vocab_size=181
max_position_embeddings=505
num_attention_heads=12
num_hidden_layers=8
type_vocab_size=1
hidden_dropout_prob=0.2
attention_probs_dropout_prob=0.2
hidden_size=768



def pre_processing_smi(smiles_str, tokenizer,smiles_maxlen=300):
    encode_dict = tokenizer.encode_plus(
        text=smiles_str,  # 输入的smiles字符串
        max_length=smiles_maxlen,  # 300
        padding='max_length',  # 300
        truncation=True)  # 超过最大长度截断处理
    smiles_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).view(1, -1)  # smiles ----> tensor
    smiles_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).view(1, -1)
    return smiles_ids, smiles_mask

class NISTDveDataset(Dataset):
    def __init__(self, data_path, mlm_probability,special_tokens_mask,prob_replace_mask,prob_replace_rand):

        if not os.path.exists(data_path):
            self.data = []
            self.tokenizer = RobertaTokenizer.from_pretrained('models/tokenizer-smiles-roberta')
            filename = "data/all_data/dev23/dev.json"
            with open(filename, 'r') as fp:  # path在main.py中source_data_path赋值 : data/all_data
                file_data = json.load(fp)
            for tmp_data in file_data:
                input_ids, attention_mask = pre_processing_smi(tmp_data["smiles"], self.tokenizer) # [smiles_ids, smiles_mask]  2维
                inputs, labels = self.mask_tokens(input_ids,mlm_probability,special_tokens_mask,prob_replace_mask,prob_replace_rand)   
    ###################### data结构 input_ids, labels, attention_mask ############################################                                                              
                self.data.append((inputs, labels, attention_mask))
                # 保存处理后的数据
            torch.save(self.data, 'data/all_data/mlm_data/dev23_mlm.pth')
        else:
            self.data = torch.load(data_path)
            self.tokenizer = RobertaTokenizer.from_pretrained('models/tokenizer-smiles-roberta')

                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
    def mask_tokens(self, inputs, mlm_probability,special_tokens_mask,prob_replace_mask,prob_replace_rand):
        """
        :param inputs: (seq_len)
        :return: (seq_len), (seq_len)
        """
        labels = inputs.clone()
        # 生成一个全为0.15 的矩阵，维度和inputs的大小一样
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # 这里的spcial_tokens指的是cls，sep,pad等，special_tokens_mask的维度和inputs一样，
        # 特殊token的位置是true，其他位置是false
        if special_tokens_mask is None:
            # 
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # masked_fill_函数的作用是会根据special_tokens_mask中true的位置，将probability_matrix中对于的位置置为0
        # 前面special_tokens_mask已经将cls等特殊字符的位置置为true,也就是通过将probability_matrix中特殊字符的位置置0，从而
        # 不会mask掉特殊字符。

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # Bert是将句子中15%的词随机mask掉，也就是每个词有15%的概率被mask掉（1），有85%的概率不被mask掉（0），
        # 因此可以建模为伯努利分布，参数p=0.15，然后通过bernoulli函数生成一个mask矩阵，维度和inputs一样，
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

class NISTDataset(Dataset):
    def __init__(self, data_end, data_path, mlm_probability,special_tokens_mask,prob_replace_mask,prob_replace_rand):

        if not os.path.exists(data_path):
            self.data = []
            self.tokenizer = RobertaTokenizer.from_pretrained('models/tokenizer-smiles-roberta')
            for i in tqdm.tqdm(range(data_end)):
                begin, end = i * 5000 + 1, (i + 1) * 5000
                filename = "data/all_data/train23/" + str(begin) + "-" + str(end) + ".json"

                with open(filename, 'r') as fp:  # path在main.py中source_data_path赋值 : data/all_data
                    file_data = json.load(fp)

                for tmp_data in file_data:
                    input_ids, attention_mask = pre_processing_smi(tmp_data["smiles"], self.tokenizer) # [smiles_ids, smiles_mask]  2维
                    inputs, labels = self.mask_tokens(input_ids,mlm_probability,special_tokens_mask,prob_replace_mask,prob_replace_rand)   
    ###################### data结构 input_ids, labels, attention_mask ############################################                                                              
                    self.data.append((inputs, labels, attention_mask))
                # 保存处理后的数据
            torch.save(self.data, 'data/all_data/mlm_data/train23_mlm.pth')
        else:
            self.data = torch.load(data_path)
            self.tokenizer = RobertaTokenizer.from_pretrained('models/tokenizer-smiles-roberta')

                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
    def mask_tokens(self, inputs, mlm_probability,special_tokens_mask,prob_replace_mask,prob_replace_rand):
        """
        :param inputs: (seq_len)
        :return: (seq_len), (seq_len)
        """
        labels = inputs.clone()
        # 生成一个全为0.15 的矩阵，维度和inputs的大小一样
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # 这里的spcial_tokens指的是cls，sep,pad等，special_tokens_mask的维度和inputs一样，
        # 特殊token的位置是true，其他位置是false
        if special_tokens_mask is None:
            # 
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # masked_fill_函数的作用是会根据special_tokens_mask中true的位置，将probability_matrix中对于的位置置为0
        # 前面special_tokens_mask已经将cls等特殊字符的位置置为true,也就是通过将probability_matrix中特殊字符的位置置0，从而
        # 不会mask掉特殊字符。

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # Bert是将句子中15%的词随机mask掉，也就是每个词有15%的概率被mask掉（1），有85%的概率不被mask掉（0），
        # 因此可以建模为伯努利分布，参数p=0.15，然后通过bernoulli函数生成一个mask矩阵，维度和inputs一样，
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

# --------------------定义模型--------------------
config = RobertaConfig(vocab_size=vocab_size,
                       max_position_embeddings=max_position_embeddings,
                       num_attention_heads=num_attention_heads,
                       num_hidden_layers=num_hidden_layers,
                       type_vocab_size=type_vocab_size,
                       hidden_dropout_prob=hidden_dropout_prob,
                       attention_probs_dropout_prob=attention_probs_dropout_prob)
roberta_mlm_model = RobertaForMaskedLM(config)
# if roberta_mlm_model.supports_gradient_checkpointing:
#     roberta_mlm_model.gradient_checkpointing_enable()
# print(sum(p.numel() for p in roberta_mlm_model.parameters()))

# 混精度训练
scaler = GradScaler()

# --------------------定义训练函数--------------------
def train(model, device, train_dataloader, dev_dataloader):
    """
    训练
    :param model: nn.Module
    :param train_dataloader: DataLoader
    """
    # assert device.startswith('cuda') or device == 'cpu', ValueError("Invalid device.")
    # device = torch.device(device)
    model_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model.to(device)
    print(len(train_dataloader))
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")
        
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
    # warm up 和 学习率衰减
    total_steps = len(train_dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5000,
                                                num_training_steps=total_steps)
    loss_dict = {}
    loss_dict['train'] = []
    loss_dict['dev'] = []
    for cur_epc in range(int(epochs)):
        training_loss = 0
        print("Epoch: {}".format(cur_epc+1))
        model.train()
        for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc='Step')):
            optimizer.zero_grad()
            input_ids, labels, attention_mask = batch
            # print(input_ids.shape, labels.shape, attention_mask.shape)
            input_ids = input_ids.squeeze(0).squeeze(1).to(device)
            labels = labels.squeeze(0).squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(0).squeeze(1).to(device)

            with autocast():
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.long()).loss
                

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            training_loss += loss.item()

        
        dev_loss = dev(model, device, dev_dataloader)
        train_loss = training_loss / len(train_dataloader)
        loss_dict['train'].append(train_loss)
        loss_dict['dev'].append(dev_loss)
        save_path = 'models/roberta_pretrained_model/'
        # torch.save(roberta_mlm_model.roberta.embeddings.state_dict(), os.path.join(save_path, 'roberta_embed_mlm' + model_suffix + '.pth'))
        # torch.save(roberta_mlm_model.roberta.encoder.state_dict(), os.path.join(save_path, 'roberta_encoder_mlm' + model_suffix + '.pth'))

        # 绘制loss曲线并保存

        plt.plot(loss_dict['train'], label='train')
        plt.plot(loss_dict['dev'], label='dev')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'loss_curve_8_' + model_suffix + '.png'))
        plt.close()
        # draw_loss(loss_dict, os.path.join(save_path, 'loss_7_' + model_suffix + '.png'))
        torch.save(roberta_mlm_model.state_dict(), os.path.join(save_path, 'roberta_mlm_8_' + model_suffix + '.pth'))
        torch.save(loss_dict, os.path.join(save_path, 'loss_record_8_' + model_suffix + '.pth'))
        
        print("learning rate:{}".format(optimizer.param_groups[0]['lr']))
        print("Training loss: ", train_loss)
        print("Dev loss: ", dev_loss)

def dev(model, device, dev_dataloader):
    """
    验证
    :param model: nn.Module
    :param dev_dataloader: DataLoader
    """
    # assert device.startswith('cuda') or device == 'cpu', ValueError("Invalid device.")
    # device = torch.device(device)
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


traindataloader = DataLoader(NISTDataset(data_end, train_data_path, mlm_probability, special_tokens_mask, prob_replace_mask, prob_replace_rand), batch_size, shuffle=True, drop_last=True)
devdataloader = DataLoader(NISTDveDataset(dev_data_path, mlm_probability, special_tokens_mask, prob_replace_mask, prob_replace_rand), batch_size, shuffle=True, drop_last=True)

train(model=roberta_mlm_model, device=device, train_dataloader=traindataloader, dev_dataloader=devdataloader)
