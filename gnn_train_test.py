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
# from bm_dataset_no_edge_attr import BMDataset
from models_graph_res import GNNModel
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#%%
def old_generate_eval_figure(data, pred):
    initial_gt = data['cloth_initial'].cpu()
    final_gt = data['cloth_final'].cpu()
    pred = pred.cpu()

    aspect = (12, 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=aspect)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    s2 = ax1.scatter(final_gt[:,0], final_gt[:,1], alpha=0.6)
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.8, 1.2])
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')

    ax2.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2)
    s3 = ax2.scatter(pred.detach()[:,0], pred.detach()[:,1], color='red', alpha=0.6)
    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.8, 1.2])
    ax2.set_xlabel('x position')
    ax2.set_ylabel('y position')
    # plt.show()

    fig.legend((s1,s2,s3), ('Initial GT', 'Final GT', 'Final Predicted'), 'lower center', ncol=3, borderaxespad=1.5)

    return fig


#%%
def generate_eval_figure(data, pred):
    initial_gt = data['cloth_initial'].cpu()

    final_gt = data['ground_truth'].cpu()
    final_gt_cloth = final_gt[:(-1-len(data['human_pose']))]
    final_gt_human = final_gt[-len(data['human_pose']):]

    pred = pred.cpu().detach()
    pred_cloth = pred[:(-1-len(data['human_pose']))]
    pred_human = pred[-len(data['human_pose']):]
    # print(pred_cloth[0])
    # print(pred_human)

    aspect = (12, 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=aspect)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    s2 = ax1.scatter(final_gt_cloth[:,0], final_gt_cloth[:,1], alpha=0.6)
    s3 = ax1.scatter(final_gt_human[:,0], final_gt_human[:,1], c=final_gt_human[:,2], marker="X", cmap='Dark2', s=90)
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.9, 1.05])
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.invert_yaxis()



    ax2.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    # s3 = ax2.scatter(pred.detach()[:,0], pred.detach()[:,1], color='red', alpha=0.6)
    s4 = ax2.scatter(pred_cloth[:,0], pred_cloth[:,1],  color='green', alpha=0.6)
    s5 = ax2.scatter(pred_human[:,0], pred_human[:,1], c=pred_human[:,2], marker="X", cmap='Set1', s=90)
    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.9, 1.05])
    ax2.set_xlabel('x position')
    ax2.set_ylabel('y position')
    ax2.invert_yaxis()
    # plt.show()

    fig.legend((s1,s2,s4,s3), ('Initial GT', 'Final GT', 'Final Pred', 'Human Points'), 'lower center', ncol=4, borderaxespad=1.5)

    return fig


#%%
#! CHECK WHY THIS EPOCH IS HERE IN YUFEI'S CODE
def run(args, epoch, dataloader, model, optimizer, scheduler, criterion, mode, device, take_images=False, fig_dir=None):
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
            #! ADD THIS WHEN RUNNING NO EDGE ATTR DATASETS
            edge_attr = torch.zeros(data['edge_index'].size()[1], 1, dtype=torch.float32, device=device)
            data['edge_attr'] = edge_attr
            # data['edge_attr'] = data['edge_attr'].float()

            state_target = data['ground_truth']
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

            if take_images:
                figure = generate_eval_figure(data, state_predicted)
                figure.savefig(osp.join(fig_dir, f'eval_{i}.png'))
                # plt.show()
                plt.close()
        
        if mode == 'train':
            scheduler.step(total_state_loss / len(dataloader))
        


    eval_metrics = {
        'total_loss':total_state_loss,
        'rmse': state_rmse_error_gt,
        'relative_error': state_relative_error_gt}

    return eval_metrics
    # return (eval_metrics, out)

#%%
def run_task(dataset, exp_dir, edge_thres, action_to_node, proc_layers, cov, sub_samp):
    torch.cuda.empty_cache()
    print(f"TEST: edge threshold = {edge_thres}, action to {action_to_node} nodes, processing layers = {proc_layers}")

    run_id = f"human_cov={cov}_et={edge_thres}cm_an={action_to_node}_pl={proc_layers}_norm=L2_sub-samp={sub_samp}_{round(time.time())}"
    writer_dir = exp_dir + run_id
    # print(writer_dir)
    writer = SummaryWriter(writer_dir)

    seed = 1001
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("CUDA AVAILABLE")
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    
    
    checkpoints_dir = osp.join(checkpoints_dir, run_id)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0")

    # dataset = dataset.shuffle()
    train_len = round(len(dataset)*0.9)
    TRAIN_DATASET = dataset[:train_len]
    TEST_DATASET = dataset[train_len:]

    # cpus = multiprocessing.cpu_count()//2
    # cpus = multiprocessing.cpu_count() - 1
    batch_size = 20
    num_workers = 8

    # TRAIN_DATASET = TEST_DATASET = [dataset[0]]
    # batch_size = num_workers = 1

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
        output_size=3,
        node_dim=7,
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
        take_images = False
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
                            take_images)
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
def evaluate(dataset, exp_dir, edge_thres, action_to_node, proc_layers, sub_samp, run_dir=None):
    print(f"TEST: edge threshold = {edge_thres}, action to {action_to_node} nodes, processing layers = {proc_layers}")

    run_id = f"et={edge_thres}cm_an={action_to_node}_pl={proc_layers}_norm=L2_sub-samp={sub_samp}_{round(time.time())}"
    # writer_dir = osp.join(exp_dir, 'eval', run_id)
    # # print(writer_dir)
    # writer = SummaryWriter(writer_dir)

    seed = 1001
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("CUDA AVAILABLE")
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # checkpoints_dir = '/home/kpputhuveetil/git/vBM-GNNdev/checkpoints'
    # checkpoints_dir = osp.join(checkpoints_dir, run_id)
    # Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0")

    # dataset = dataset.shuffle()
    train_len = round(len(dataset)*0.9)
    TRAIN_DATASET = dataset[:train_len]
    TEST_DATASET = dataset[train_len:]

    # cpus = multiprocessing.cpu_count()//2
    # cpus = multiprocessing.cpu_count() - 1
    batch_size = 10
    num_workers = 8


    # TRAIN_DATASET = TEST_DATASET = [dataset[0]]
    # batch_size = num_workers = 1


    # trainDataLoader = torch_geometric.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                                                 pin_memory=True, drop_last=True)
    testDataLoader = torch_geometric.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                    pin_memory=True, drop_last=True)
    imageDataLoader = torch_geometric.data.DataLoader(TEST_DATASET[0:100], batch_size=1, shuffle=False, num_workers=1,
                                                    pin_memory=True, drop_last=True)

    print("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))

    args = SimpleNamespace(
        seed=1001,
        learning_rate=1e-4,
        epoch = 250,
        proc_layer_num=proc_layers, 
        global_size=0,
        output_size=3,
        node_dim=8,
        edge_dim=1)
    checkpoint_path = '/home/kpputhuveetil/git/vBM-GNNdev/checkpoints/noedge_human_cov=False_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643620595'
    # checkpoint_path = '/home/kpputhuveetil/git/vBM-GNNdev/checkpoints/noedge_human_cov=True_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643615404'
    # checkpoint_path = '/home/kpputhuveetil/git/vBM-GNNdev/checkpoints/wgtedge_human_cov=False_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643697045'
    # checkpoint_path = '/home/kpputhuveetil/git/vBM-GNNdev/checkpoints/wgtedge_human_cov=True_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643734996'
    checkpoint_path = osp.join(checkpoint_path, 'model_249.pth')

    run_dir = "/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/runs/human_no_edge_attr/human_cov=False_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643620595"
    # run_dir = "/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/runs/human_no_edge_attr/human_cov=True_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643615404"
    # run_dir = '/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/runs/human_edge_attr/human_cov=False_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643697045'
    # run_dir = "/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/runs/human_edge_attr/human_cov=True_et=6cm_an=ALL_pl=4_norm=L2_sub-samp=True_1643734996"
    fig_dir = osp.join(run_dir, 'images')
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # checkpoint_path = '/home/kpputhuveetil/git/vBM-GNNdev/checkpoints/et=3cm_an=GRASP_pl=4_norm=L2_sub-samp=True_1642405730_overfit/model_249.pth'
    model = GNNModel(args, args.proc_layer_num, args.global_size, args.output_size)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.to(device)
    model_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    train_metrics = run(
                        args, 
                        None, 
                        testDataLoader, 
                        model, 
                        optimizer, 
                        scheduler,
                        model_criterion,
                        'eval', 
                        device,
                        False)
    print(train_metrics)

    # train_metrics = run(
    #                     args, 
    #                     None, 
    #                     imageDataLoader, 
    #                     model, 
    #                     optimizer, 
    #                     scheduler,
    #                     model_criterion,
    #                     'eval', 
    #                     device,
    #                     True,
    #                     fig_dir)
    # print(train_metrics)

    # print(train_metrics)
    torch.cuda.empty_cache()


    print('DONE!\n')

#%%
# torch.cuda.empty_cache()
# dataset = BMDataset(root='data_2089', proc_data_dir='humanALL_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
# run_task(dataset, 'runs/human_no_edge_attr/',6, 'ALL', 4, False, True)
# #%%
# dataset = BMDataset(root='data_2089', proc_data_dir='humanCOV_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
# run_task(dataset, 'runs/human_no_edge_attr/',6, 'ALL', 4, True, True)

#%%
# torch.cuda.empty_cache()
# dataset = BMDataset(root='data_2089', proc_data_dir='wgt_humanALLedgeattr_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
# run_task(dataset, 'runs/human_edge_attr/',6, 'ALL', 4, False, True)
# #%%
# dataset = BMDataset(root='data_2089', proc_data_dir='wgt_humanCOVedgeattr_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
# run_task(dataset, 'runs/human_edge_attr/',6, 'ALL', 4, True, True)


# %%
dataset = BMDataset(root='data_2089', proc_data_dir='humanALL_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
evaluate(dataset, '',6, 'ALL', 4,False, True)
# %%
dataset = BMDataset(root='data_2089', proc_data_dir='humanALL_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
evaluate(dataset, '',6, 'ALL', 4,True, True)


# %%
dataset = BMDataset(root='data_2089', proc_data_dir='wgt_humanALLedgeattr_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
evaluate(dataset, '',6, 'ALL', 4,False, True)
# %%
dataset = BMDataset(root='data_2089', proc_data_dir='wgt_humanCOVedgeattr_sub-samp_edge-thres=6cm_action=ALL', subsample = True)
evaluate(dataset, '',6, 'ALL', 4,True, True)