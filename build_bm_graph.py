#%%
from cgi import test
import glob, pickle, multiprocessing, math, os
from logging import root
import os.path as osp
import numpy as np
from numpy.lib import index_tricks
import pandas as pd

import torch, torch_geometric
from torch_geometric.data import Dataset, Data

from gnn_train_test_new import GNN_Train_Test

#%%


class BM_Graph():
    def __init__(self, voxel_size=float('NaN'), edge_threshold=0.06, action_to_all=True, cloth_initial=None):
        self.voxel_size = voxel_size
        self.subsample = True if not (np.isnan(self.voxel_size)) else False
        self.edge_threshold = edge_threshold
        self.action_to_all = action_to_all
        # self.testing = testing

        if self.subsample:
            initial_blanket_state_3D = self.sub_sample_point_clouds(cloth_initial)
        self.initial_blanket_2D = np.delete(np.array(initial_blanket_state_3D), 2, axis = 1)
        self.edge_indices = self.get_edge_connectivity(initial_blanket_state_3D)
        self.edge_features = torch.zeros(self.edge_indices.size()[0], 1, dtype=torch.float)
        self.global_vector = torch.zeros(1, 0, dtype=torch.float32)


    def build_graph(self, action):

        node_features = self.get_node_features(self.initial_blanket_2D, action)

        data = Data(
            x = node_features,
            edge_attr = self.edge_features,
            edge_index = self.edge_indices.t().contiguous(),
            batch = torch.zeros(node_features.shape[0], dtype=torch.long)
        )

        return data
    
    def get_node_features(self, cloth_initial_2D_pos, action):
        """
        returns an array with shape (# nodes, node feature size)
        convert list of lists to tensor
        """
        scale = [0.44, 1.05]*2
        action_scaled = action*scale

        if self.action_to_all:
            nodes = np.append(cloth_initial_2D_pos, [action]*len(cloth_initial_2D_pos), axis=1).tolist()

        return torch.tensor(nodes, dtype=torch.float)
    
    def get_edge_connectivity(self, cloth_initial):
        """
        returns an array of edge indexes, returned as a list of index tuples
        Data requires indexes to be in COO format so will need to convert via performing transpose (.t()) and calling contiguous (.contiguous())
        """
        cloth_initial_3D = np.array(cloth_initial)
        cloth_initial_2D = np.delete(cloth_initial_3D, 2, axis = 1)
        threshold = self.edge_threshold
        edge_inds = []
        for p1_ind, point_1 in enumerate(cloth_initial_2D):
            for p2_ind, point_2 in enumerate(cloth_initial_2D): # want duplicate edges to capture both directions of info sharing
                if p1_ind != p2_ind and np.linalg.norm(point_1 - point_2) <= threshold: # don't consider distance between a point and itself, see if distance is within
                    edge_inds.append([p1_ind, p2_ind])
                np.linalg.norm(point_1 - point_2) <= threshold
        # return torch.tensor([0,2], dtype = torch.long)
        return torch.tensor(edge_inds, dtype = torch.long)

    def sub_sample_point_clouds(self, cloth_initial_3D_pos):
        cloth_initial = np.array(cloth_initial_3D_pos)
        voxel_size = self.voxel_size
        nb_vox=np.ceil((np.max(cloth_initial, axis=0) - np.min(cloth_initial, axis=0))/voxel_size)
        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((cloth_initial - np.min(cloth_initial, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted=np.argsort(inverse)

        voxel_grid={}
        voxel_grid_cloth_inds={}
        cloth_initial_subsample=[]
        last_seen=0
        for idx,vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)]= cloth_initial[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
            voxel_grid_cloth_inds[tuple(vox)] = idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]
            
            closest_point_to_barycenter = np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()
            cloth_initial_subsample.append(voxel_grid[tuple(vox)][closest_point_to_barycenter])

            last_seen+=nb_pts_per_voxel[idx]

        return cloth_initial_subsample

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


# #%%
# f = '/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/data_2089/raw/c0_331897332481794079_pid15959.pkl'
# raw_data = pickle.load(open(f, "rb"))
# cloth_initial = raw_data['info']['cloth_initial'][1]
# graph = BM_Graph(
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     cloth_initial=cloth_initial)
# #%%
# action = raw_data['action']
# graph_d = graph.build_graph(action)
# # %%
# from models_graph_res import GNNModel
# from types import SimpleNamespace

# epochs = 300
# proc_layers = 8
# learning_rate = 1e-4
# seed = 1001
# global_size = 0
# output_size = 2
# node_dim = 6
# edge_dim = 1

# args = SimpleNamespace(
#             seed=seed,
#             learning_rate=learning_rate,
#             epoch = epochs,
#             proc_layer_num=proc_layers, 
#             global_size=global_size,
#             output_size=output_size,
#             node_dim=node_dim,
#             edge_dim=edge_dim)

# checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/test/checkpoints'
# checkpoint_path = osp.join(checkpoint, 'model_249.pth')

# model = GNNModel(args, args.proc_layer_num, args.global_size, args.output_size)
# model.load_state_dict(torch.load(checkpoint_path)['model'])
# model.eval()
# # model.to(self.device)
# # # %%
# # d = torch_geometric.data.DataLoader([d], batch_size=1, shuffle=False, num_workers=1,
# # #                                                             pin_memory=True, drop_last=True)


# #%%

# # without dataloader
# data = graph_d.to_dict()
# # batch = torch.zeros(graph_d.x.shape[0], dtype=torch.float32)
# batch = data['batch']
# print(batch)
# print(batch.shape)
# batch_num = np.max(batch.data.cpu().numpy()) + 1
# print(batch_num)
# global_vec = torch.zeros(int(batch_num), args.global_size, dtype=torch.float32)
# data['u'] = global_vec
# pred = model(data)['target']
# #%%

# # with dataloader


# d = torch_geometric.data.DataLoader([graph_d], batch_size=1, shuffle=False, num_workers=1,
#                                                           pin_memory=True, drop_last=True)
# for i, data in enumerate(d):
#     data = data.to_dict()
#     batch = data['batch']
#     print(batch)
#     print(batch.shape)
#     batch_num = np.max(batch.data.cpu().numpy()) + 1
#     print(batch_num)
#     global_vec = torch.zeros(batch_num, args.global_size, dtype=torch.float32)
#     print(global_vec)
#     data['u'] = global_vec
#     pred = model(data)['target']
# # %%
# print(pred[0])
# print(pred.shape)
# # %%
