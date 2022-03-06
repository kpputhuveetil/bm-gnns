#%%
# from _typeshed import NoneType
import os, multiprocessing
import os.path as osp
import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric
from torch_geometric.data import Data
import time
from pathlib import Path

# from bm_dataset import BMDataset
from bm_dataset_3D import BMDataset
# from bm_dataset_no_edge_attr import BMDataset
from models_graph_res import GNNModel
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import configparser

#%%
class GNN_Train_Test():
    def __init__(self, save_dir, cuda_device, dataset=None, train_test=None, proc_layers=None, num_images=None, epochs=None, learning_rate=None, seed=None, batch_size=None, num_workers=None, model_descirption=None):
        torch.cuda.empty_cache()
        # print(f"TEST: edge threshold = {edge_thres}, action to {action_to_node} nodes, processing layers = {proc_layers}")
        self.model_id = f"{model_descirption}_epochs={epochs}_batch={batch_size}_workers={num_workers}_{round(time.time())}"
        self.writer_dir = osp.join(save_dir, self.model_id, 'runs')
        self.writer = SummaryWriter(self.writer_dir)

        self.checkpoints_dir = osp.join(save_dir, self.model_id, 'checkpoints')
        Path(self.checkpoints_dir).mkdir(parents=True, exist_ok=True)

        self.config_dir = osp.join(save_dir, self.model_id, 'config.ini')

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(cuda_device)
        print("CUDA Device:", self.device)

        # dataset = dataset.shuffle()

        # if you pass a single float for train_test, interpret as the ratio of train:test points to use from the entire dataset
        if isinstance(train_test, float):
            train_test_ratio = train_test
            train_len = round(len(dataset)*train_test_ratio)
            test_len = -1
        # if you pass a tuple for train_test, interpret as the number of train points and test points to select from the entire dataset of points
        elif isinstance(train_test, tuple): 
            train_len = train_test[0]
            test_len = train_test[0] + train_test[1]
        TRAIN_DATASET = dataset[:train_len]
        TEST_DATASET = dataset[train_len:test_len]
        # TEST_DATASET = dataset[-1* train_test[1]:]

        # cpus = multiprocessing.cpu_count()//2
        # cpus = multiprocessing.cpu_count() - 1
        # batch_size = 100
        # num_workers = 8
        self.batch_size = batch_size
        self.num_workers = num_workers

        # TRAIN_DATASET = TEST_DATASET = [dataset[0]]
        # batch_size = num_workers = 1
        if dataset is not None:
            self.trainDataLoader = torch_geometric.data.DataLoader(TRAIN_DATASET, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                                            pin_memory=True, drop_last=True)
            self.testDataLoader = torch_geometric.data.DataLoader(TEST_DATASET, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                                            pin_memory=True, drop_last=True)
            self.imageDataLoader = torch_geometric.data.DataLoader(TEST_DATASET[0:num_images], batch_size=1, shuffle=False, num_workers=1,
                                                            pin_memory=True, drop_last=True)

            print("The number of training data is: %d" % len(TRAIN_DATASET))
            print("The number of test data is: %d" % len(TEST_DATASET))
            print("The number of image data is: %d" % num_images)

            node_dim = dataset[0].x.shape[1]
            edge_dim = dataset[0].edge_attr.shape[1]
            output_size = dataset[0].cloth_final.shape[1]
            global_size = 0 #! Don't Hardcode

        print(f"Node feature length: {node_dim}, edge feature length: {edge_dim}, global_size: {global_size}, output size: {output_size}")
        print("Processing layers:", proc_layers)

        self.args = SimpleNamespace(
            seed=seed,
            learning_rate=learning_rate,
            epoch = epochs,
            proc_layer_num=proc_layers, 
            global_size=global_size,
            output_size=output_size,
            node_dim=node_dim,
            edge_dim=edge_dim)
        self.model = GNNModel(self.args, self.args.proc_layer_num, self.args.global_size, self.args.output_size)
        self.model.to(self.device)
        self.model_criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=3, verbose=True)

        config = configparser.ConfigParser()
        config['DEFAULT'] = {}
        config['Dataset'] = {
            'dir':dataset.root,
            'train_test':train_test,
            'num_total_data':len(dataset),
            'num_train_data':len(TRAIN_DATASET),
            'num_test_data':len(TEST_DATASET),
            'voxel_size':dataset.voxel_size,
            'subsample':dataset.subsample,
            'edge_threshold':dataset.edge_threshold,
            'action_to_all':dataset.action_to_all}
        config['Model'] = {
            'seed':self.args.seed,
            'learning_rate':self.args.learning_rate,
            'epochs': self.args.epoch,
            'proc_layer_num':self.args.proc_layer_num, 
            'global_size':self.args.global_size,
            'output_size':self.args.output_size,
            'node_dim':self.args.node_dim,
            'edge_dim':self.args.edge_dim}
        with open(self.config_dir, 'w') as configfile:
            config.write(configfile)
        print("Set up complete")
    
    def run(self, args, epoch, dataloader, mode, take_images=False, fig_dir=None):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()

        total_state_loss = 0
        state_relative_error_gt = 0
        state_rmse_error_gt = 0


        with torch.set_grad_enabled(mode == 'train'):
            for i, data in enumerate(dataloader):
                if mode == 'train':
                    self.optimizer.zero_grad()

                data = data.to(self.device).to_dict()
                batch = data['batch']
                batch_num = np.max(batch.data.detach().cpu().numpy()) + 1
                global_vec = torch.zeros(batch_num, args.global_size, dtype=torch.float32, device=self.device)
                data['u'] = global_vec
                #! ADD THIS WHEN RUNNING NO EDGE ATTR DATASETS
                # edge_attr = torch.zeros(data['edge_index'].size()[1], 1, dtype=torch.float32, device=self.device)
                # data['edge_attr'] = edge_attr
                # data['edge_attr'] = data['edge_attr'].float()

                state_target = data['cloth_final']
                state_predicted = self.model(data)['target']
                # out = (state_target, state_predicted)

                # L1 norm for MSE
                # state_predicted = state_predicted.contiguous().view(-1, 1)
                # state_target = state_target.contiguous().view(-1, 1)
                
                # L2 norm for MSE
                state_predicted = state_predicted.contiguous()
                state_target = state_target.contiguous()

                state_loss = self.model_criterion(state_predicted, state_target)
                
                
                total_state_loss += state_loss.detach().item()

                state_pred_numpy = state_predicted.data.detach().cpu().numpy().flatten()
                state_target_numpy = state_target.data.detach().cpu().numpy().flatten()

                state_relative_error_gt += np.mean(np.abs(state_pred_numpy - state_target_numpy) / (np.abs(state_target_numpy) + 1e-10))
                state_rmse_error_gt += np.sqrt(np.mean((state_pred_numpy - state_target_numpy) ** 2))

                if mode == 'train':
                    state_loss.backward()
                    self.optimizer.step()
                
                total_state_loss += state_loss.item()

                if take_images:
                    figure = self.old_generate_eval_figure(data, state_predicted)
                    figure.savefig(osp.join(fig_dir, f'eval_{i}.png'))
                    # plt.show()
                    plt.close()
            
            if mode == 'train':
                self.scheduler.step(total_state_loss / len(dataloader))
            


        eval_metrics = {
            'total_loss':total_state_loss,
            'rmse': state_rmse_error_gt,
            'relative_error': state_relative_error_gt}

        return eval_metrics
        # return (eval_metrics, out)

    def train(self):
        best_loss = best_rmse = best_rele = best_time = None

        t_initial = time.time()
        for epoch in tqdm(range(self.args.epoch)):
            t0 = time.time()
            take_images = False
            train_metrics = self.run(
                                    self.args, 
                                    epoch, 
                                    self.trainDataLoader, 
                                    'train',
                                    take_images=False)
            t1 = time.time()
            self.writer.add_scalar("Loss/train", train_metrics['total_loss'], epoch)
            self.writer.add_scalar("RMSE/train", train_metrics['rmse'], epoch)
            self.writer.add_scalar("Relative_error/train", train_metrics['relative_error'], epoch)
            self.writer.add_scalar("Time_per_epoch/train", t1-t0, epoch)

            if (best_loss is None) or (train_metrics['total_loss'] < best_loss):
                best_loss = train_metrics['total_loss']
                best_rmse = train_metrics['rmse']
                best_rele = train_metrics['relative_error']
                best_time = t1-t0

            # print(train_metrics)
            torch.cuda.empty_cache()

            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            savepath = osp.join(self.checkpoints_dir, 'model_{}.pth'.format(epoch))
            torch.save(save_dict, savepath)

        self.writer.add_hparams(
        {'proc_layers': self.args.proc_layer_num},
        {  
            'best_loss':best_loss,
            'best_rmse':best_rmse,
            'best_rele':best_rele,
            'best_time':best_time
        },
        run_name='hparams'
        )
        self.writer.flush()
        self.writer.close()

        print('Training Done!\n')
    
    def evaluate(self, checkpoint_path):
        checkpoint_path = osp.join(checkpoint_path, 'model_249.pth')

        self.model.load_state_dict(torch.load(checkpoint_path)['model'])
        self.model.to(self.device)

        run_dir = "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/test/runs"
        fig_dir = osp.join(run_dir, 'images')
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        eval_metrics = self.run(
                                self.args, 
                                None, 
                                self.testDataLoader, 
                                'eval',
                                take_images=False)
        print(eval_metrics)
        torch.cuda.empty_cache()
        print('Evaluation Done!\n')
    
    def evaluate_images(self, checkpoint_path, run_dir):
        checkpoint_path = osp.join(checkpoint_path, 'model_249.pth')

        self.model.load_state_dict(torch.load(checkpoint_path)['model'])
        self.model.to(self.device)

        fig_dir = osp.join(run_dir, 'images')
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        eval_metrics = self.run(
                                self.args, 
                                None, 
                                self.imageDataLoader, 
                                'eval',
                                take_images=True,
                                fig_dir=fig_dir)
        print(eval_metrics)
        torch.cuda.empty_cache()
    
    def old_generate_eval_figure(self, data, pred):
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


# %%
# dataset = BMDataset(
#     root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_new_data/3D_dataset', 
#     description='50k_samples_3D',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)
dataset = BMDataset(
    root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_new_data/3D_dataset', 
    description='50k_samples_3D',
    voxel_size=0.05, 
    edge_threshold=0.07, 
    action_to_all=True, 
    testing=False)
save_dir = osp.abspath(os.path.join(os.getcwd(), os.pardir, 'trained_models'))
model_description = 'train10k_3D'
cuda_device = 'cuda:0'
train_test = (10000, 100)
num_images = 100
epochs = 250
proc_layers = 4
learning_rate = 1e-4
seed = 1001
batch_size = 100
num_workers = 4

gnn_train_test = GNN_Train_Test(
    save_dir, cuda_device, dataset, train_test, 
    proc_layers, num_images, epochs, learning_rate, seed, batch_size, num_workers,
    model_description)

#%%
gnn_train_test.train()

# #%%
# checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_epochs=250_batch=100_workers=4_1646202554/checkpoints'
# # gnn_train_test.evaluate(gnn_train_test.checkpoints_dir)
# # gnn_train_test.evaluate(checkpoint)

# checkpoint2 = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/50ktest/checkpoints'
# # gnn_train_test.evaluate(checkpoint2)

# checkpoint3 = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/9ktest/checkpoints'

#%%
model_path = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_3D_epochs=250_batch=100_workers=4_1646464130'
gnn_train_test.evaluate(osp.join(model_path,"checkpoints"))
gnn_train_test.evaluate_images(osp.join(model_path,"checkpoints"), osp.join(model_path,"runs"))

#%%
# gnn_train_test.evaluate_images(checkpoint, "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_epochs=250_batch=100_workers=4_1646202554/runs")
# gnn_train_test.evaluate_images(checkpoint2, "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/50ktest/runs")
# gnn_train_test.evaluate_images(checkpoint3, "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/test/runs")

# %%
# gnn_train_test.evaluate_images(checkpoint3, "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/test/runs")