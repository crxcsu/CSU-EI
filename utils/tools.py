import os
import shutil
import random
import numpy as np
import torch
import pandas as pd

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
    os.makedirs(LOG_DIR + '/mainlib_all_uni_feature', exist_ok=True)
    os.makedirs(LOG_DIR + '/plot_result_analyzer', exist_ok=True)


def copy_folder(src, dst):
    for folders, subfolders, filenames in os.walk(src):
        for filename in filenames:
            shutil.copy(os.path.join(folders, filename), dst)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    print(f"Random seed set to {seed}")


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