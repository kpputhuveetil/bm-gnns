#%%
from audioop import mul
import sys, time, math, os
from torch import multiprocessing

import os

from sklearn.metrics import coverage_error
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs, get_reward, scale_action, check_grasp_on_cloth, randomize_target_limbs
# import assistive_gym.envs.bu_gnn_util
import numpy as np
import cma
import pickle
import os.path as osp
from build_bm_graph import BM_Graph
import torch
from tqdm import tqdm
from gnn_train_test_new import GNN_Train_Test
from torch_geometric.data import Dataset, Data
from bm_dataset import BMDataset
# sys.stderr = stderr
#%%

def counter_callback(output):
    return
    print()

#%%
def access_model(idx, model, data, device):

    data = data.to(device).to_dict()

    batch = torch.zeros(data['x'].shape[0], dtype=torch.long)
    data['batch'] = batch
    
    batch_num = np.max(batch.data.cpu().numpy()) + 1
    global_size = 0
    global_vec = torch.zeros(int(batch_num), global_size, dtype=torch.float32, device=device)
    # global_vec = torch.zeros(int(batch_num), global_size, dtype=torch.float32)
    data['u'] = global_vec

    pred = model(data)['target'].detach().numpy()

    return pred



#%%
#! START MAIN
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')


    # * make the enviornment, set the specified target limb code and an initial seed value
    checkpoint = "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_cont_learn_epochs=250_batch=100_workers=4_1646202554"

    dataset = BMDataset(
        root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_blanket_var_data', 
        description='blanket_var',
        voxel_size=0.05, 
        edge_threshold=0.06, 
        action_to_all=True, 
        testing=False)

    device = 'cpu'
    gnn_train_test = GNN_Train_Test(device)
    gnn_train_test.load_model_from_checkpoint(checkpoint)
    gnn_train_test.model.share_memory()
    gnn_train_test.model.eval()

    data = dataset[0]
    model = gnn_train_test.model

    t0 = time.time()

    result_objs = []

    num_processes = 10
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i in range(num_processes):
            result = pool.apply_async(access_model, args=(i, model, data, device))
            result_objs.append(result)
        
        results = [result.get() for result in result_objs]
    t1 = time.time()
    # print(results)
    print(t1-t0)
