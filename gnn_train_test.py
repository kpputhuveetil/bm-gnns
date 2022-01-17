#%%
# from _typeshed import NoneType
import os, multiprocessing
import os.path as osp
import numpy as np
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric
from torch_geometric.data import Data
import time
from pathlib import Path

from bm_dataset import BMDataset
from models_graph_res import GNNModel
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#%%
def generate_eval_figure(dataloader, pred):
    initial_gt = dataloader[0].cloth_initial
    final_gt = dataloader[0].cloth_final

    aspect = (10, 8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=aspect)
    ax1.scatter(initial_gt[:,0], initial_gt[:,1])
    ax1.scatter(final_gt[:,0], final_gt[:,1])
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.7, 1.1])

    ax2.scatter(initial_gt[:,0], initial_gt[:,1])
    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.7, 1.1])
    ax2.scatter(pred.detach()[:,0], pred.detach()[:,1], color='red')
    # plt.show()

    return fig


#%%
def run(args, epoch, dataloader, model, optimizer, scheduler, criterion, mode, device, save_visual_path=None):
    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()

    total_state_loss = 0
    state_relative_error_gt = 0
    state_rmse_error_gt = 0


    with torch.set_grad_enabled(mode == 'train'):
        for i, data in enumerate(dataloader):
            if mode == 'train':
                optimizer.zero_grad()

            data = data.to(device).to_dict()
            batch = data['batch']
            batch_num = np.max(batch.data.cpu().numpy()) + 1
            global_vec = torch.zeros(batch_num, args.global_size, dtype=torch.float32, device=device)
            data['u'] = global_vec
            edge_attr = torch.zeros(data['edge_index'].size()[1], 1, dtype=torch.float32, device=device)
            data['edge_attr'] = edge_attr

            state_target = data['cloth_final']
            state_predicted = model(data)['target']
            # out = (state_target, state_predicted)

            # L1 norm for MSE
            # state_predicted = state_predicted.contiguous().view(-1, 1)
            # state_target = state_target.contiguous().view(-1, 1)
            
            # L2 norm for MSE
            state_predicted = state_predicted.contiguous()
            state_target = state_target.contiguous()

            state_loss = criterion(state_predicted, state_target)
            
            
            total_state_loss += state_loss.item()

            state_pred_numpy = state_predicted.data.cpu().numpy().flatten()
            state_target_numpy = state_target.data.cpu().numpy().flatten()

            state_relative_error_gt += np.mean(np.abs(state_pred_numpy - state_target_numpy) / (np.abs(state_target_numpy) + 1e-10))
            state_rmse_error_gt += np.sqrt(np.mean((state_pred_numpy - state_target_numpy) ** 2))

            if mode == 'train':
                state_loss.backward()
                optimizer.step()
            
            total_state_loss += state_loss.item()
        
        if mode == 'train':
            scheduler.step(total_state_loss / len(dataloader))

    eval_metrics = {
        'total_loss':total_state_loss,
        'rmse': state_rmse_error_gt,
        'relative_error': state_relative_error_gt}

    return eval_metrics
    # return (eval_metrics, out)

#%%
def run_task(dataset, exp_dir, edge_thres, action_to_node, proc_layers, sub_samp):
    print(f"TEST: edge threshold = {edge_thres}, action to {action_to_node} nodes, processing layers = {proc_layers}")

    run_id = f"et={edge_thres}cm_an={action_to_node}_pl={proc_layers}_norm=L2_sub-samp={sub_samp}_{round(time.time())}"
    writer_dir = exp_dir + run_id
    # print(writer_dir)
    writer = SummaryWriter(writer_dir)

    seed = 1001
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    checkpoints_dir = '/home/kpputhuveetil/git/bm_gnns/checkpoints'
    checkpoints_dir = osp.join(checkpoints_dir, run_id)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)


    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = dataset.shuffle()
    train_len = round(len(dataset)*0.9)
    TRAIN_DATASET = dataset[:train_len]
    TEST_DATASET = dataset[train_len:]

    cpus = multiprocessing.cpu_count()//2
    cpus = multiprocessing.cpu_count() - 1
    batch_size = cpus
    num_workers = cpus
    trainDataLoader = torch_geometric.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                    pin_memory=True, drop_last=True)
    testDataLoader = torch_geometric.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                    pin_memory=True, drop_last=True)

    print("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))

    args = SimpleNamespace(
        seed=1001,
        learning_rate=1e-4,
        epoch = 250,
        proc_layer_num=proc_layers, 
        global_size=0,
        output_size=2,
        node_dim=6,
        edge_dim=1)
    model = GNNModel(args, args.proc_layer_num, args.global_size, args.output_size)
    model.to(device)
    model_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    best_loss = best_rmse = best_rele = best_time = None

    t_initial = time.time()
    for epoch in tqdm(range(args.epoch)):
        t0 = time.time()
        save_visual_path = None
        train_metrics = run(
                            args, 
                            epoch, 
                            trainDataLoader, 
                            model, 
                            optimizer, 
                            scheduler,
                            model_criterion,
                            'train', 
                            device,
                            save_visual_path)
        t1 = time.time()
        writer.add_scalar("Loss/train", train_metrics['total_loss'], epoch)
        writer.add_scalar("RMSE/train", train_metrics['rmse'], epoch)
        writer.add_scalar("Relative_error/train", train_metrics['relative_error'], epoch)
        writer.add_scalar("Time_per_epoch/train", t1-t0, epoch)

        if (best_loss is None) or (train_metrics['total_loss'] < best_loss):
            best_loss = train_metrics['total_loss']
            best_rmse = train_metrics['rmse']
            best_rele = train_metrics['relative_error']
            best_time = t1-t0

        # print(train_metrics)
        torch.cuda.empty_cache()


        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        savepath = osp.join(checkpoints_dir, 'model_{}.pth'.format(epoch))
        torch.save(save_dict, savepath)

    writer.add_hparams(
    {
        'edge_threshold': edge_thres,
        'action_to_node': action_to_node,
        'proc_layers': proc_layers
    },
    {  
        'best_loss':best_loss,
        'best_rmse':best_rmse,
        'best_rele':best_rele,
        'best_time':best_time
    },
    run_name='hparams'
    )
    writer.flush()
    writer.close()



    print('DONE!\n')

#%%
dataset = BMDataset(root='data_2089', proc_data_dir='sub-samp_edge-thres=3cm_action=GRASP', subsample = True)
run_task(dataset, 'runs/sub_samp/',3, 'GRASP', 4, True)
# %%
