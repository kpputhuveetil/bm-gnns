#%%
from cgi import test
import glob, pickle, multiprocessing, math, os, sys
from logging import root
import os.path as osp
import numpy as np
from numpy.lib import index_tricks
import pandas as pd

import torch, torch_geometric
from torch_geometric.data import Dataset, Data

from tqdm import tqdm

# sys.path.insert(1, '/home/kpputhuveetil/git/vBM-GNNs/assistive-gym-fem/assistive_gym/envs')
# from bu_gnn_util import sub_sample_point_clouds


#!!! DELETE CHANGES FOR CMA-DATA
#%%


class BMDataset(Dataset):
    def __init__(self, root, description, transform=None, pre_transform=None, voxel_size=float('NaN'), edge_threshold=0.06, action_to_all=True, use_cma_data=False, testing=False):
        """
        root is where the data should be stored (data). The directory is split into "raw" and 'preprocessed' directories
        raw folder contains original pickle files
        prprocessed will be filled with the dataset when this class is instantiated
        """
        self.voxel_size = voxel_size
        self.subsample = True if not (np.isnan(self.voxel_size)) else False
        self.edge_threshold = edge_threshold
        self.action_to_all = action_to_all
        self.testing = testing

        path = os.getcwd()
        data_dir = osp.join(path, root, 'raw/*.pkl')
        #voxel size and edge threshold in cm
        proc_data_dir = f"{description}_vs{self.voxel_size}-et{self.edge_threshold}-aa{int(self.action_to_all)}"
        root = osp.join(root, proc_data_dir)
        # print(root)

        self.filenames_raw = glob.glob(data_dir)
        # print(self.filenames_raw)
        self.file_count = len(self.filenames_raw)
        self.num_processes =  multiprocessing.cpu_count()-1
        self.reps = math.ceil(self.file_count/self.num_processes)

        if self.file_count%self.num_processes != 0:
            buffer = [None]*(self.num_processes - (self.file_count%self.num_processes))
            self.filenames = self.filenames_raw + buffer
        else:
            self.filenames = self.filenames_raw

        # # FOR TESTING
        if self.testing:
            self.filenames = self.filenames[0]
            self.num_processes =1
            self.reps = 1
        
        self.unpickling_errors = []
        self.use_cma_data = use_cma_data
        super(BMDataset, self).__init__(root, transform, pre_transform)


    @property
    def raw_file_names(self):
        """
        if files exists in raw directory, download is not triggered (download function also not implemented here)
        """
        return self.filenames_raw

    @property
    def processed_file_names(self):
        """
        If these files are found in processed, processing is skipped (don't need to start from scratch)
        Not implemented here
        """
        """ If these files are found in processed_dir, processing is skipped"""
        proc_files = [f'data_{i}.pt' for i in range(len(self.filenames_raw))]
        # print(proc_files[0:10])
        # print(len(proc_files))
        # return [f'data_{i}.pt' for i in range(10)]
        return [f'data_{i}.pt' for i in range(len(self.filenames_raw))]
        # return glob.glob(r'/home/kpputhuveetil/git/bm_gnns/data/processed/*.pt')


    def download(self):
        """
        not implemented
        """
        # # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        # ...
        pass

    def process(self):
        """
        Allows us to construct a graph and pass it to a Data object (which models a single graph) for each data file
        """
        # self.filenames = self.filenames[0:self.num_processes]
        files_array = np.reshape(self.filenames, (self.reps, self.num_processes))
        result_objs = []

        print(self.processed_dir)

        for rep, files in enumerate(tqdm(files_array)):
            # print(f"Rep: {rep+1}, Total Processed: {rep*self.num_processes}")
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                for i,f in enumerate(files):
                    result = pool.apply_async(self.build_graph, args = [f, i+(rep*127)])
                    result_objs.append(result)

                results = [result.get() for result in result_objs]
        
        print("Processing Complete!")


    def build_graph(self, f, idx):
        # global idx
        # data = Data()
        # if not self.testing:
        #     torch.save(data, osp.join(self.processed_dir, f'data_test{idx}.pt'))
        # return


        #! TODO: CHECK THIS!!
        if f is None:
            return

        try:
            raw_data = pickle.load(open(f, "rb"))
        except:
            print('UnpickingError:', f)
            self.unpickling_errors.append(f)
            return 
        
        action = raw_data['action']
        if self.use_cma_data:
            raw_data = raw_data['sim_info']
        human_pose = raw_data['observation'][0]
       

        # initial_num_cloth_points = raw_data['info']['cloth_initial'][0]
        initial_blanket_state = raw_data['info']['cloth_initial'][1]

        # final_num_cloth_points = raw_data['info']['cloth_final'][0]
        final_blanket_state = raw_data['info']['cloth_final'][1]

        if self.subsample:
            initial_blanket_state, final_blanket_state = self.sub_sample_point_clouds(initial_blanket_state, final_blanket_state)

        # Get  node features
        node_features = self.get_node_features(initial_blanket_state, action)
        
        edge_indices = self.get_edge_connectivity(initial_blanket_state)
    
        edge_features = torch.zeros(edge_indices.size()[0], 1, dtype=torch.float)
        cloth_initial, cloth_final = self.get_cloth_as_tensor(initial_blanket_state, final_blanket_state)

        # Read data from `raw_path`.
        data = Data(
            x = node_features,
            edge_attr = edge_features,
            edge_index = edge_indices.t().contiguous(),
            cloth_initial = cloth_initial,
            cloth_final = cloth_final,
            action = torch.tensor(action, dtype=torch.float),
            human_pose = torch.tensor(human_pose, dtype=torch.float)
        )

        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        if not self.testing:
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    
    def get_node_features(self, cloth_initial, action):
        """
        returns an array with shape (# nodes, node feature size)
        convert list of lists to tensor
        """
        scale = [0.44, 1.05]*2
        action_scaled = action*scale


        if self.action_to_all:
            ## ACTION TO ALL CLOTH POINTS
            nodes = []
            for ind, point in enumerate(cloth_initial):
                node_feature = list(point[0:2]) + list(action_scaled)
                nodes.append(node_feature)
                #! USE SOMETHING LIKE THIS INSTEAD
                # nodes = np.append(cloth_initial_3D_pos, [action]*len(cloth_initial_3D_pos), axis=1).tolist()

        else:
            # ACTION ONLY TO GRASPED CLOTH POINTS
            grasp_loc = action_scaled[0:2]
            dist = []
            for i, v in enumerate(cloth_initial):
                v = np.array(v)
                d = np.linalg.norm(v[0:2] - grasp_loc)
                dist.append(d)
            anchor_idx = np.argpartition(np.array(dist), 4)[:4]

            nodes = []
            for ind, point in enumerate(cloth_initial):
                if ind in anchor_idx:
                    node_feature = list(point[0:2]) + list(action_scaled)
                else:
                    node_feature = list(point[0:2]) + [0]*len(action_scaled)
                nodes.append(node_feature)

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

    def get_cloth_as_tensor(self, cloth_initial_3D_pos, cloth_final_3D_pos):
        cloth_initial_2D_pos = np.delete(np.array(cloth_initial_3D_pos), 2, axis = 1)
        cloth_final_2D_pos = np.delete(np.array(cloth_final_3D_pos), 2, axis = 1)

        cloth_i_tensor = torch.tensor(cloth_initial_2D_pos, dtype=torch.float)
        cloth_f_tensor = torch.tensor(cloth_final_2D_pos, dtype=torch.float)
        return cloth_i_tensor, cloth_f_tensor

    def sub_sample_point_clouds(self, cloth_initial_3D_pos, cloth_final_3D_pos):
        cloth_initial = np.array(cloth_initial_3D_pos)
        cloth_final = np.array(cloth_final_3D_pos)
        voxel_size = self.voxel_size
        nb_vox=np.ceil((np.max(cloth_initial, axis=0) - np.min(cloth_initial, axis=0))/voxel_size)
        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((cloth_initial - np.min(cloth_initial, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted=np.argsort(inverse)

        voxel_grid={}
        voxel_grid_cloth_inds={}
        cloth_initial_subsample=[]
        cloth_final_subsample = []
        last_seen=0
        for idx,vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)]= cloth_initial[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
            voxel_grid_cloth_inds[tuple(vox)] = idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]
            
            closest_point_to_barycenter = np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()
            cloth_initial_subsample.append(voxel_grid[tuple(vox)][closest_point_to_barycenter])
            cloth_final_subsample.append(cloth_final[voxel_grid_cloth_inds[tuple(vox)][closest_point_to_barycenter]])

            last_seen+=nb_pts_per_voxel[idx]

        return cloth_initial_subsample, cloth_final_subsample

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# #%%        
# dataset = BMDataset(
#     root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_blanket_var_data', 
#     description='blanket_var',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)

# dataset = BMDataset(
#     root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_high_pose_var_data', 
#     description='50k_samples',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)

#%%        
# dataset = BMDataset(
#     root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_new_data', 
#     description='50k_samples',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)

#%%
# dataset = BMDataset(
#     root='data_2089', 
#     description='dyn-gnn',
#     voxel_size=0.05, 
#     edge_threshold=0.06, 
#     action_to_all=True, 
#     testing=False)
#%%
# dataset = BMDataset(root='data_2089', proc_data_dir='edge-thres=2cm_action=GRASP')
# dataset_sub = BMDataset(root='data_2089', proc_data_dir='sub-samp_edge-thres=6cm_action=ALL', subsample = True)
# # print(dataset)
# print(dataset_sub)



# #%%
# ind = 0
# print(dataset[ind])
# print(dataset[ind].edge_index.t().size())
# print(dataset[ind].x.size())
# # print(dataset[ind].y.size())

# len(dataset)
# #%%
# ind = 0
# print(dataset_sub[ind])
# print(dataset_sub[ind].edge_index.t().size())
# print(dataset_sub[ind].x.size())
# # print(dataset[ind].y.size())

# len(dataset_sub)
# #%%
# ratio = []
# for point in dataset:
#     edges = point.edge_index.t().size()[0]
#     nodes = point.x.size()[0]
#     ratio.append(nodes/edges)
# print(np.mean(ratio), np.std(ratio))


# #%%
# print(dataset[ind].x[100])
# # %%
# # %%

# import matplotlib.pyplot as plt
# import numpy as np

# dataset = BMDataset(root='data/')
# # # print(dataset)

# ind = 0

# data_i = dataset[ind].cloth_initial
# x_i = data_i[:,0]
# y_i = data_i[:,1]

# data_f = dataset[ind].cloth_final
# x_f = data_f[:,0]
# y_f = data_f[:,1]

# plt.scatter(x_i, y_i)
# plt.scatter(x_f, y_f)
# plt.show()
# %%