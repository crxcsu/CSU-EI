import os
import shutil
import random
import numpy as np
import torch

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

