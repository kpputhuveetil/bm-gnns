#%%
import gym, sys, argparse, multiprocessing, time, os
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
from assistive_gym.envs.bu_gnn_util import get_body_points_from_obs, get_body_points_reward
from gym.utils import seeding
import numpy as np
import cma
from cma.optimization_tools import EvalParallel2
import pickle
import pathlib
import os.path as osp
from gnn_train_test_new import GNN_Train_Test
from build_bm_graph import BM_Graph
import torch, torch_geometric
from torch_geometric.data import Dataset, Data
from models_graph_res import GNNModel
from types import SimpleNamespace
import assistive_gym
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob

#%%
def grasp_on_cloth(action, cloth_initial):
    grasp_loc = action[0:2]

    dist = []
    for i, v in enumerate(cloth_initial):
        v = np.array(v)
        d = np.linalg.norm(v[0:2] - grasp_loc)
        dist.append(d)
    # * if no points on the blanket are within 2.8 cm of the grasp location, exit 
    return (np.any(np.array(dist) < 0.028))

def cost_function(action, all_body_points):
    scale = [0.44, 1.05]*2
    action = scale*action

    if not grasp_on_cloth(action, cloth_initial):
        return [0, -1, -1]

    data = graph.build_graph(action)
    data = data.to_dict()
    batch = data['batch']
    batch_num = np.max(batch.data.cpu().numpy()) + 1
    global_size = 0
    global_vec = torch.zeros(int(batch_num), global_size, dtype=torch.float32)
    data['u'] = global_vec
    pred = model(data)['target'].detach().numpy()
    # print('predicted', pred[0:10])
    cost, covered_status = get_cost(action, all_body_points, graph.initial_blanket_2D, pred)

    return [cost, pred, covered_status]

def get_cost(action, all_body_points, cloth_initial_2D, cloth_final_2D):
    reward_distance_btw_grasp_release = -150 if np.linalg.norm(action[0:2] - action[2:]) >= 1.5 else 0
    body_point_reward, covered_status = get_body_points_reward(all_body_points, cloth_initial_2D, cloth_final_2D)
    reward = body_point_reward + reward_distance_btw_grasp_release
    cost = -reward
    return cost, covered_status


def generate_figure(cloth_initial, pred, human_pose, all_body_points, covered_status):
    cloth_final = pred
    point_colors = []

    for point in covered_status:
        is_target = point[0]
        is_covered = point[1]
        if is_target == 1:
            color = 'purple' if is_covered else 'forestgreen'
        elif is_target == -1:
            color = 'red' if is_covered else 'darkorange'
        else:
            color = 'darkorange' if is_covered else 'red'
        point_colors.append(color)

    aspect = (4, 6)

    fig, ax1 = plt.subplots(figsize=aspect)

    ax1.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)
    ax1.scatter(human_pose[:,0], human_pose[:,1], c='navy')

    ntarg = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='uncovered points')
    targ = mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='None', label='target points')
    obs = mlines.Line2D([], [], color='navy', marker='o', linestyle='None', label='observation points')
    ax1.scatter(cloth_final[:,0], cloth_final[:,1], alpha=0.2)
    ax1.axis([-0.7, 0.7, -1.0, 0.9])
    # plt.legend(loc='lower left', prop={'size': 9}, handles=[ntarg, targ, obs])
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.invert_yaxis()
    plt.show()




    
    # initial_gt = np.array(cloth_initial)

    # aspect = (4, 6)

    # fig, ax1 = plt.subplots(figsize=aspect)
    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(1.0)

    # s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    # s2 = ax1.scatter(pred[:,0], pred[:,1], alpha=0.6)
    # s3 = ax1.scatter(human_pose[:,0], human_pose[:,1], c=covered_status, marker="X", cmap='Dark2', s=90)
    # ax1.set_xlim([-0.7, 0.7])
    # ax1.set_ylim([-0.8, 1.2])
    # ax1.set_xlabel('x position')
    # ax1.set_ylabel('y position')
    # ax1.invert_yaxis()

    # plt.show()

    # fig.legend((s1,s2,s3), ('Initial GT', 'Final Predicted', 'Human'), 'lower center', ncol=3, borderaxespad=0.3)

    return fig

def load_model(checkpoint):
    checkpoint_path = osp.join(checkpoint, 'model_249.pth')
    epochs = 250
    proc_layers = 4
    learning_rate = 1e-4
    seed = 1001
    global_size = 0
    output_size = 2
    node_dim = 6
    edge_dim = 1

    args = SimpleNamespace(
                seed=seed,
                learning_rate=learning_rate,
                epoch = epochs,
                proc_layer_num=proc_layers, 
                global_size=global_size,
                output_size=output_size,
                node_dim=node_dim,
                edge_dim=edge_dim)

    model = GNNModel(args, args.proc_layer_num, args.global_size, args.output_size)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()

    return model

#%%
#! START MAIN

# * make the enviornment, set the specified target limb code and an initial seed value
checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/50ktest/checkpoints'
model = load_model(checkpoint)

# * set the number of processes to 1/4 of the total number of cpus
# *     collect data for 4 different target limbs simultaneously by running this script in 4 terminals
num_proc = multiprocessing.cpu_count()//16
num_proc = 8

data_dir = "/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/data_2089"
data_dir = osp.join(data_dir, 'raw/*.pkl')
filenames = glob.glob(data_dir)
eval_files = filenames[-1:]
target_limb_code = 14
# print(filenames)

for i, f in enumerate(eval_files):
    print(i)

    # cloth_initial, human_pose = env.reset()
    # print(cloth_initial)
    # f = '/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/data_2089/raw/c0_331897332481794079_pid15959.pkl'
    raw_data = pickle.load(open(f, "rb"))
    cloth_initial = raw_data['info']['cloth_initial'][1]
    # human_pose = raw_data['observation'][0]
    human_pose = np.reshape(raw_data['observation'][0], (-1,2))
    all_body_points = get_body_points_from_obs(human_pose, target_limb_code=target_limb_code)
    

    # print('captured cloth initial')
    graph = BM_Graph(
        voxel_size=0.05, 
        edge_threshold=0.06, 
        action_to_all=True, 
        cloth_initial=cloth_initial)

    # print('graph constructed')

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': num_proc, 'maxfevals': 300, 'tolfun': 1e-2, 'tolflatfitness': 10, 'tolfunhist': 1e-20}) # , 'tolfun': 10, 'maxfevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)
    x0 = [0.5, 0.5, -0.5, -0.5]
    # x0 = np.random.uniform(-1,1,4)
    sigma0 = 0.2
    reward_threshold = 95

    pose_count = 1
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
        fevals += num_proc
        total_fevals += num_proc
        
        actions = es.ask()
        output = [cost_function(x, all_body_points) for x in actions]
        t1 = time.time()
        output = [list(x) for x in zip(*output)]
        costs = output[0]
        preds = output[1]
        covered_status = output[2]
        print(-1*np.array(costs))
        es.tell(actions, costs)
        
        if (best_cost is None) or (np.min(costs) < best_cost):
            best_cost = np.min(costs)
            best_cost_ind = np.argmin(costs)
            best_reward = -best_cost
            best_action = actions[best_cost_ind]
            best_pred = preds[best_cost_ind]
            best_covered_status = covered_status[best_cost_ind]
            best_time = t1 - t0
        if best_reward >= 95:
            break

        # # * if any of the processes reached the reward_threshold, stop optimizing
        # if np.any(np.array(costs) <= -reward_threshold):
        #     print("Reward threshold reached")
        #     break
        # if fevals >= 300:
        #     print("No solution found after 300 fevals")
        #     break
    # print(best_cost)
    # es.result_pretty()

    # env.set_seed_val(seeding.create_seed())
    # env.set_target_limb_code(target_limb)


    # fig_dir = '/home/kpputhuveetil/git/vBM-GNNdev/cmaes_images'
    # fig = generate_figure(cloth_initial, best_pred, human_pose, best_covered_status)
    fig_name = f'eval{i}_time={best_time:.2f}_rew={best_reward:.2f}_tl={target_limb_code}.png'
    # fig.savefig(osp.join(fig_dir, fig_name))
    print(fig_name, best_action)

# %%
fig = generate_figure(cloth_initial, best_pred, human_pose, all_body_points, best_covered_status)
# %%
