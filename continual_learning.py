#%%
import sys, time, math, os
# import multiprocessing
from torch import multiprocessing

from sklearn.metrics import coverage_error
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
from assistive_gym.envs.bu_gnn_util import *
# import assistive_gym.envs.bu_gnn_util
from cma_gnn_util import *

import numpy as np
import cma
import pickle
import os.path as osp
from build_bm_graph import BM_Graph
import torch
from models_graph_res import GNNModel
from types import SimpleNamespace
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from assistive_gym.learn import make_env
from gym.utils import seeding
from pathlib import Path
from heapq import nlargest
from tqdm import tqdm
from gnn_train_test_new import GNN_Train_Test
from torch_geometric.data import Dataset, Data

from bm_dataset import BMDataset

#%%
# check if grasp is on the cloth BEFORE subsampling! cloth_initial_raw is pre subsampling
def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

def cost_function(action, all_body_points, cloth_initial_raw, graph, model, device):
    action = scale_action(action)
    cloth_initial = graph.initial_blanket_2D
    if not grasp_on_cloth(action, cloth_initial_raw):
        return [0, cloth_initial, -1, None]

    data = graph.build_graph(action)

    data = data.to(device).to_dict()
    batch = data['batch']
    batch_num = np.max(batch.data.cpu().numpy()) + 1
    # batch_num = np.max(batch.data.detach().cpu().numpy()) + 1    # version used for gpu, not cpu only for this script
    global_size = 0
    global_vec = torch.zeros(int(batch_num), global_size, dtype=torch.float32, device=device)
    data['u'] = global_vec


    pred = model(data)['target'].detach().numpy()
    # print('predicted', pred[0:10])
    cost, covered_status = get_cost(action, all_body_points, cloth_initial, pred)

    return [cost, pred, covered_status, data]

def get_cost(action, all_body_points, cloth_initial_2D, cloth_final_2D):
    reward, covered_status = get_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)
    cost = -reward
    return cost, covered_status

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: CMA-ES Best Reward:{output[1]:.2f}, Sim Reward: {output[2]:.2f}, CMA Time: {output[3]/60:.2f}, TL: {output[4]}")
    
    # print(f"{counter} - Trial Completed: {output[0]}, Worker: {output[2]}, Filename: {output[1]}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")

#%%
def gnn_cma(env_name, idx, model, device, target_limb_code, seed, iter_data_dir):

    coop = 'Human' in env_name
    # seed = seeding.create_seed()
    env = make_env(env_name, coop=coop, seed=seed)
    done = False
    # #env.render())
    human_pose = env.reset()
    human_pose = np.reshape(human_pose, (-1,2))
    if target_limb_code is None:
        target_limb_code = randomize_target_limbs()
    cloth_initial_raw = env.get_cloth_state()
    env.set_target_limb_code(target_limb_code)
    pop_size = 8
    # return seed, env.target_limb_code, human_pose[1, 0], idx

    # f = eval_files[0]
    # raw_data = pickle.load(open(f, "rb"))
    # cloth_initial = raw_data['info']['cloth_initial'][1]
    # human_pose = raw_data['observation'][0]
    # human_pose = np.reshape(raw_data['observation'][0], (-1,2))
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code)
    

    # print('captured cloth initial')
    graph = BM_Graph(
        root = iter_data_dir,
        description=f"iter_{iter}_processed",
        voxel_size=0.05, 
        edge_threshold=0.06, 
        action_to_all=True, 
        cloth_initial=cloth_initial_raw)

    # print('graph constructed')

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': pop_size, 'maxfevals': 300, 'tolfun': 1e-2, 'tolflatfitness': 10, 'tolfunhist': 1e-20}) # , 'tolfun': 10, 'maxfevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)
    x0 = [0.5, 0.5, -0.5, -0.5]
    # x0 = np.random.uniform(-1,1,4)
    sigma0 = 0.2
    reward_threshold = 95

    total_fevals = 0

    fevals = 0
    iterations = 0
    t0 = time.time()

    # * initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    # print('Env and CMA-ES set up')

    # # * evaluate cost function in parallel over num_cpus/4 processes
    # with EvalParallel2(cost_function, number_of_processes=num_proc) as eval_all:
    # * continue running optimization until termination criteria is reached
    best_cost = None
    while not es.stop():
        iterations += 1
        fevals += pop_size
        total_fevals += pop_size
        
        actions = es.ask()
        output = [cost_function(x, all_body_points, cloth_initial_raw, graph, model, device) for x in actions]
        t1 = time.time()
        output = [list(x) for x in zip(*output)]
        costs = output[0]
        preds = output[1]
        covered_status = output[2]
        data = output[3]
        # print(-1*np.array(costs))
        es.tell(actions, costs)
        
        if (best_cost is None) or (np.min(costs) < best_cost):
            best_cost = np.min(costs)
            best_cost_ind = np.argmin(costs)
            best_reward = -best_cost
            best_action = actions[best_cost_ind]
            best_pred = preds[best_cost_ind]
            best_covered_status = covered_status[best_cost_ind]
            best_data = data[best_cost_ind]
            best_time = t1 - t0
            best_fevals = fevals
            best_iterations = iterations
        if best_reward >= reward_threshold:
            break
    observation, env_reward, done, info = env.step(best_action)     
    # print(info.keys())

    # return cloth_initial, best_pred, all_body_points, best_covered_status, info
    sim_info = {'observation':observation, 'reward':env_reward, 'done':done, 'info':info}
    cma_info = {
        'best_cost':best_cost, 'best_reward':best_reward, 'best_pred':best_pred, 'best_time':best_time,
        'best_covered_status':best_covered_status, 'best_fevals':best_fevals, 'best_iterations':best_iterations}
    
    save_data_to_pickle(
        idx, 
        seed, 
        best_action, 
        human_pose, 
        target_limb_code,
        sim_info,
        cma_info,
        iter_data_dir,
        best_covered_status)
    # save_dataset(idx, graph, best_data, sim_info, best_action, human_pose, best_covered_status)
    return seed, best_reward, env_reward, best_time, target_limb_code


def load_model_for_eval(checkpoint):
    pass

def load_model_for_update():
    pass

def evaluate_dyn_model(target_limb_code, trials, model, seeds, iter_data_dir, device, num_processes):

    result_objs = []
    for j in tqdm(range(math.ceil(trials/num_processes))):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                idx = i+(j*num_processes)
                result = pool.apply_async(gnn_cma, args = (env_name, idx, model, device, target_limb_code, seeds[i], iter_data_dir), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            all_results.append(results)
    
    results_array = np.array(results)
    pred_sim_reward_error = abs(results_array[:,2] - results_array[:,1])
    largest_error_inds = np.argpartition(pred_sim_reward_error, k_largest)[-k_largest:]
    eval_order_seeds = list(zip(*results))[0]

    return pred_sim_reward_error, largest_error_inds, eval_order_seeds


def update_dyn_model(model):
    pass

#%%
#! START MAIN
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    env_name = "BodiesUncoveredGNN-v1"
    target_limb_code = None


    checkpoint = "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_cont_learn_epochs=250_batch=100_workers=4_1646202554"
    data_dir = osp.join(checkpoint, 'cont_learning_data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    tracking_file = open(osp.join(data_dir, f'tracking_{round(time.time())}.txt'), 'a')

    initial_dataset = BMDataset(
        root='/home/kpputhuveetil/git/vBM-GNNdev/gnn_new_data', 
        description='50k_samples',
        voxel_size=0.05, 
        edge_threshold=0.06, 
        action_to_all=True, 
        testing=False)
    initial_dataset = initial_dataset[:10100]

    device = 'cpu'
    gnn_train_test = GNN_Train_Test(device)
    gnn_train_test.load_model_from_checkpoint(checkpoint)
    gnn_train_test.set_initial_dataset(initial_dataset, (10000, 100))
    gnn_train_test.set_dataloaders(100)

    counter = 0

    # reserve one cpu to keep working while collecting data
    num_processes = multiprocessing.cpu_count() - 1

    # num data points to collect
    start_iter = 0
    iterations = 100
    seeds = [[]]*(iterations+1)
    
    trials = 100
    num_processes = 50
    k_largest = int(trials/2)

    start_iter = 13
    checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_cont_learn_epochs=250_batch=100_workers=4_1646202554/cont_learning_data/iteration_12'

    seeds[0] = [seeding.create_seed() for i in range(trials)]

    # print(filenames)
    all_results = []
    for iter in tqdm(range(iterations)):
        iter_data_dir = osp.join(data_dir, f"iteration_{iter}")
        Path(iter_data_dir).mkdir(parents=True, exist_ok=True)


        
        gnn_train_test.set_new_save_dir(iter_data_dir)
        tracking_file.write(f"----Iteration {iter}----\n")
        tracking_file.write(f"Checkpoint: {checkpoint}\n")
        tracking_file.flush()

        gnn_train_test.load_model(checkpoint)
        gnn_train_test.model.to(torch.device('cpu'))
        gnn_train_test.model.share_memory()
        gnn_train_test.model.eval()

        pred_sim_reward_error, largest_error_inds, eval_order_seeds = evaluate_dyn_model(target_limb_code, trials, gnn_train_test.model, seeds[iter], iter_data_dir, device, num_processes)
        recheck_seeds = [eval_order_seeds[ind] for ind in largest_error_inds]  # index seeds in this way to prevent float conversion
        new_seeds = [seeding.create_seed() for i in range(k_largest)]
        seeds[iter+1] = recheck_seeds + new_seeds

        print(f"Eval Model Iteration {iter} - Ave Pred-Sim Error {np.mean(pred_sim_reward_error)}")
        print(pred_sim_reward_error)
        tracking_file.write(f"Eval Modesl Iteration {iter} - Ave Pred-Sim Error {np.mean(pred_sim_reward_error)}\n")
        tracking_file.flush()
        # print(seeds)
        gnn_train_test.delete_model()


        print('begin training on new data')
        tracking_file.write(f"Training\n")
        tracking_file.flush()

        gnn_train_test.set_device('cuda:0')
        gnn_train_test.load_model(checkpoint)
        new_dataset = BMDataset(
            root= iter_data_dir, 
            description=f'cont_learn_data_{iter}',
            voxel_size=0.05, 
            edge_threshold=0.06, 
            action_to_all=True, 
            testing=False)
        gnn_train_test.add_to_train_set(new_dataset)
        gnn_train_test.set_dataloaders(100)
        gnn_train_test.train(20)

        # torch.distributed.barrier()
        torch.cuda.synchronize()
        checkpoint = iter_data_dir

    tracking_file.close()
    #     dataset = BMDataset(
    #         root= iter_data_dir, 
    #         description=f'cont_learn_data_{iter}',
    #         voxel_size=0.05, 
    #         edge_threshold=0.06, 
    #         action_to_all=True, 
    #         testing=False)




        # print(len(results))

    # print(results)
    #%%
    # # results = [result.get() for result in result_objs]
    # results_array = np.array(results)
    # print(np.mean(results_array[:,0]), np.std(results_array[:,0]))
    # print(np.mean(results_array[:,1]), np.std(results_array[:,1]))
    # print(np.mean(results_array[:,2]/60), np.std(results_array[:,2]/60))


    # %%
    # fig = generate_figure(cloth_initial, best_pred, human_pose, all_body_points, best_covered_status)
    # %%
    # import glob

    # filenames = glob.glob('/home/kpputhuveetil/git/vBM-GNNdev/cmaes_eval_data/*.pkl')
    # print(len(filenames))
    # # %%
    # for filename in filenames:
    #     with open(filename, 'rb') as f:
    #         raw_data = pickle.load(f)
    #         # raw_data["action"]
    #         # raw_data["human_pose"]
    #         # raw_data["figure": fig,
    #         targ = raw_data['target_limb_code']
    #         # raw_data['sim_info']
    #         print(targ)
    #         cov_stat = raw_data['cma_info']['best_covered_status']
    #         if targ in [6]:
    #             print('val', cov_stat)
    #         else:
    #             print('length', len(cov_stat))


    # # %%
    # filenames[2]
    # with open(filename, 'rb') as f:
    #     raw_data = pickle.load(f)
    #     action = raw_data["action"]
    #     action = scale_action(action)
    #     cloth_initial_raw = raw_data["sim_info"]["info"]['cloth_initial'][1]
    #     graph = BM_Graph(
    #         voxel_size=0.05, 
    #         edge_threshold=0.06, 
    #         action_to_all=True, 
    #         cloth_initial=cloth_initial_raw)
    #     all_body_points = get_body_points_from_obs(raw_data["human_pose"], target_limb_code=raw_data['target_limb_code'])
    #     cloth_initial = graph.initial_blanket_2D
    #     covered_status = get_covered_status(all_body_points, cloth_initial)
    #     info = raw_data['sim_info']['info']
    #     info["covered_status_sim"] = 
    #     final_sim = np.array(info["cloth_final_subsample"])
    #     print(grasp_on_cloth(action, graph.initial_blanket_2D))
    #     # generate_figure_sim_results(cloth_initial, pred, all_body_points, covered_status, info, cma_reward, sim_reward):
    #     generate_figure_sim_results(cloth_initial, cloth_initial, all_body_points, covered_status, info, cma_reward, sim_reward)
        
        

    #     # raw_data["human_pose"]
    #     # raw_data["figure": fig,
    #     # targ = raw_data['target_limb_code']
    #     # # raw_data['sim_info']
    #     # print(targ)
    #     cov_stat = raw_data['cma_info']['best_covered_status']
    #     print(cov_stat)
    #     # if targ in [6]:
    #     #     print('val', cov_stat)
    #     # else:
    #     #     print('length', len(cov_stat))
    # %%

