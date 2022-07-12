#%%
import sys, time, math, os
# import multiprocessing
from torch import multiprocessing

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

#%%
# check if grasp is on the cloth BEFORE subsampling! cloth_initial_raw is pre subsampling
def grasp_on_cloth(action, cloth_initial_raw):
    dist, is_on_cloth = check_grasp_on_cloth(action, np.array(cloth_initial_raw))
    return is_on_cloth

def cost_function(action, all_body_points, cloth_initial_raw, graph, model):
    action = scale_action(action)
    cloth_initial = graph.initial_blanket_2D
    if not grasp_on_cloth(action, cloth_initial_raw):
        return [0, cloth_initial, -1, None]

    data = graph.build_graph(action)

    data = data.to_dict()
    batch = data['batch']
    batch_num = np.max(batch.data.cpu().numpy()) + 1
    global_size = 0
    global_vec = torch.zeros(int(batch_num), global_size, dtype=torch.float32)
    data['u'] = global_vec

    #! PASS DEVICE AS PARAM
    # # GPU ver
    # device = 'cuda:0'
    # data = data.to(device).to_dict()
    # batch = data['batch']
    # batch_num = np.max(batch.data.detach().cpu().numpy()) + 1
    # global_size = 0
    # global_vec = torch.zeros(batch_num, global_size, dtype=torch.float32, device=device)
    # data['u'] = global_vec


    pred = model(data)['target'].detach().numpy()
    # print('predicted', pred[0:10])
    cost, covered_status = get_cost(action, all_body_points, cloth_initial, pred)

    return [cost, pred, covered_status, data]

def get_cost(action, all_body_points, cloth_initial_2D, cloth_final_2D):
    reward, covered_status = get_reward(action, all_body_points, cloth_initial_2D, cloth_final_2D)
    cost = -reward
    return cost, covered_status

def generate_figure_sim_results(cloth_initial, pred, all_body_points, covered_status, info, cma_reward, sim_reward):
    
    # handle case if action was clipped - there is essentially no point in generating this figure since initial and final is the same
    if isinstance(covered_status, int) and covered_status == -1:
        fig = None # clipped
        return fig

    initial_gt = np.array(cloth_initial)
    covered_status_sim = info["covered_status_sim"]
    final_sim = np.array(info["cloth_final_subsample"])

    point_colors = []
    point_colors_sim = []

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

    for point in covered_status_sim:
        is_target = point[0]
        is_covered = point[1]
        if is_target == 1:
            color = 'purple' if is_covered else 'forestgreen'
        elif is_target == -1:
            color = 'red' if is_covered else 'darkorange'
        else:
            color = 'darkorange' if is_covered else 'red'
        point_colors_sim.append(color)

    # aspect = (4, 6)
    aspect = (12, 10)

    fig, (ax2, ax1) = plt.subplots(1, 2,figsize=aspect)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    ax1.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)

    s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none', label='cloth initial')
    s3 = ax1.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)
    s2 = ax1.scatter(pred[:,0], pred[:,1], alpha=0.6, color='mediumvioletred', label='cloth final')
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.9, 1.05])
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.invert_yaxis()
    ax1.set_title(f"Predicted: Reward = {cma_reward:.2f}")

    final_sim = np.array(info["cloth_final_subsample"])

    ax2.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    # s3 = ax2.scatter(pred.detach()[:,0], pred.detach()[:,1], color='red', alpha=0.6)
    ax2.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors_sim)
    s4 = ax2.scatter(final_sim[:,0], final_sim[:,1],  color='mediumvioletred', alpha=0.6)
    
    ntarg = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='uncovered points')
    targ = mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='None', label='target points')


    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.9, 1.05])
    ax2.set_xlabel('x position')
    ax2.set_ylabel('y position')
    ax2.set_title(f"Ground Truth: Reward = {sim_reward:.2f}")
    ax2.invert_yaxis()
    # plt.show()

    # plt.show()

    # fig.legend((s1,s2,s3,s4), ('Initial GT', 'Final Predicted', 'Human', 'Final Sim'), 'lower center', ncol=4, borderaxespad=0.3)
    fig.legend(loc='lower center', handles=[ntarg, targ, s1, s2], ncol=4, borderaxespad=2)

    return fig

# def load_model(checkpoint):
#     checkpoint_path = osp.join(checkpoint, 'model_249.pth')
#     epochs = 250
#     proc_layers = 4
#     learning_rate = 1e-4
#     seed = 1001
#     global_size = 0
#     output_size = 2
#     node_dim = 6
#     edge_dim = 1

#     args = SimpleNamespace(
#                 seed=seed,
#                 learning_rate=learning_rate,
#                 epoch = epochs,
#                 proc_layer_num=proc_layers, 
#                 global_size=global_size,
#                 output_size=output_size,
#                 node_dim=node_dim,
#                 edge_dim=edge_dim)

#     model = GNNModel(args, args.proc_layer_num, args.global_size, args.output_size)
#     model.load_state_dict(torch.load(checkpoint_path)['model'])
#     model.eval()

#    return model

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: CMA-ES Best Reward:{output[1]:.2f}, Sim Reward: {output[2]:.2f}, CMA Time: {output[3]/60:.2f}, TL: {output[4]}")
    # print(f"{counter} - Trial Completed: {output[0]}, Worker: {output[2]}, Filename: {output[1]}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")

#%%
def gnn_cma(env_name, idx, model, target_limb_code, seed, iter_data_dir):

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
        output = [cost_function(x, all_body_points, cloth_initial_raw, graph, model) for x in actions]
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
    
    # save_data_to_pickle(
    #     idx, 
    #     seed, 
    #     best_action, 
    #     human_pose, 
    #     target_limb_code,
    #     sim_info,
    #     cma_info,
    #     iter_data_dir)
    # save_dataset(idx, graph, best_data, sim_info, best_action, human_pose)
    return seed, best_reward, env_reward, best_time, target_limb_code


def save_data_to_pickle(idx, seed, action, human_pose, target_limb_code, sim_info, cma_info, iter_data_dir):
    pid = os.getpid()
    filename = f"tl{target_limb_code}_c{idx}_{seed}_pid{pid}"

    raw_dir = osp.join(iter_data_dir, 'raw')
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    pkl_loc = raw_dir

    with open(os.path.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "action":action,
            "human_pose":human_pose, 
            'target_limb_code':target_limb_code,
            'sim_info':sim_info,
            'cma_info':cma_info}, f)

def save_dataset(idx, graph, data, sim_info, action, human_pose):

    initial_blanket_state = sim_info['info']["cloth_initial_subsample"]
    final_blanket_state = sim_info['info']["cloth_final_subsample"]
    cloth_initial, cloth_final = graph.get_cloth_as_tensor(initial_blanket_state, final_blanket_state)

    data['cloth_initial'] = cloth_initial
    data['cloth_final'] = cloth_final
    data['action'] = torch.tensor(action, dtype=torch.float)
    data['human_pose'] = torch.tensor(human_pose, dtype=torch.float)
    
    proc_data_dir = graph.proc_data_dir
    data = graph.dict_to_Data(data)
    torch.save(data, osp.join(proc_data_dir, f'data_{idx}.pt'))

def generate_figure_sim_results(cloth_initial, pred, all_body_points, covered_status, info, cma_reward, sim_reward):
    
    # handle case if action was clipped - there is essentially no point in generating this figure since initial and final is the same
    if isinstance(covered_status, int) and covered_status == -1:
        fig = None # clipped
        return fig

    initial_gt = np.array(cloth_initial)
    covered_status_sim = info["covered_status_sim"]
    final_sim = np.array(info["cloth_final_subsample"])

    point_colors = []
    point_colors_sim = []

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

    for point in covered_status_sim:
        is_target = point[0]
        is_covered = point[1]
        if is_target == 1:
            color = 'purple' if is_covered else 'forestgreen'
        elif is_target == -1:
            color = 'red' if is_covered else 'darkorange'
        else:
            color = 'darkorange' if is_covered else 'red'
        point_colors_sim.append(color)

    # aspect = (4, 6)
    aspect = (12, 10)

    fig, (ax2, ax1) = plt.subplots(1, 2,figsize=aspect)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    ax1.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)

    s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none', label='cloth initial')
    s3 = ax1.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)
    s2 = ax1.scatter(pred[:,0], pred[:,1], alpha=0.6, color='mediumvioletred', label='cloth final')
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.9, 1.05])
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.invert_yaxis()
    ax1.set_title(f"Predicted: Reward = {cma_reward:.2f}")

    final_sim = np.array(info["cloth_final_subsample"])

    ax2.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    # s3 = ax2.scatter(pred.detach()[:,0], pred.detach()[:,1], color='red', alpha=0.6)
    ax2.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors_sim)
    s4 = ax2.scatter(final_sim[:,0], final_sim[:,1],  color='mediumvioletred', alpha=0.6)
    
    ntarg = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='uncovered points')
    targ = mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='None', label='target points')


    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.9, 1.05])
    ax2.set_xlabel('x position')
    ax2.set_ylabel('y position')
    ax2.set_title(f"Ground Truth: Reward = {sim_reward:.2f}")
    ax2.invert_yaxis()
    # plt.show()

    # plt.show()

    # fig.legend((s1,s2,s3,s4), ('Initial GT', 'Final Predicted', 'Human', 'Final Sim'), 'lower center', ncol=4, borderaxespad=0.3)
    fig.legend(loc='lower center', handles=[ntarg, targ, s1, s2], ncol=4, borderaxespad=2)

    return fig

#%%
#! START MAIN
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    env_name = "BodiesUncoveredGNN-v1"
    # * make the enviornment, set the specified target limb code and an initial seed value
    # checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/50ktest/checkpoints'
    # checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/test/checkpoints'
    # checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_epochs=250_batch=100_workers=4_1646202554/checkpoints'
    # checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_3D_epochs=250_batch=100_workers=4_1646468311/checkpoints'
    # checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/high_pose_var_10k_epochs=250_batch=100_workers=4_1647288217/checkpoints'
    # checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/trained_with_cmaes_data_epochs=250_batch=100_workers=4_1650226144/checkpoints'

    checkpoint = "/home/kpputhuveetil/git/vBM-GNNdev/trained_models/train10k_cont_learn_epochs=250_batch=100_workers=4_1646202554"
    data_dir = osp.join(checkpoint, 'cont_learning_data')
    Path(data_dir).mkdir(parents=True, exist_ok=True)


    gnn_train_test = GNN_Train_Test('cpu')
    gnn_train_test.load_model_from_checkpoint(checkpoint)
    gnn_train_test.model.share_memory()
    gnn_train_test.model.eval()
    model = gnn_train_test.model

    # model = gnn_train_test.model
    # * set the number of processes to 1/4 of the total number of cpus
    # *     collect data for 4 different target limbs simultaneously by running this script in 4 terminals
    # num_proc = multiprocessing.cpu_count()//16
    # num_proc = 8

    # target_limb_code = 14
    target_limb_code = None

    counter = 0

    # reserve one cpu to keep working while collecting data
    num_processes = multiprocessing.cpu_count() - 1

    # num data points to collect
    iterations = 1
    seeds = [[]]*(iterations+1)


    trials = 50
    k_largest = int(trials/2)
    num_processes = 50

    seeds[0] = [seeding.create_seed() for i in range(trials)]

    # print(filenames)
    all_results = []
    for iter in tqdm(range(iterations)):
        iter_data_dir = osp.join(data_dir, f"iteration_{iter}")
        Path(iter_data_dir).mkdir(parents=True, exist_ok=True)

        result_objs = []
        for j in tqdm(range(math.ceil(trials/num_processes))):
            with multiprocessing.Pool(processes=num_processes) as pool:
                for i in range(num_processes):
                    idx = i+(j*num_processes)
                    result = pool.apply_async(gnn_cma, args = (env_name, idx, model, target_limb_code, seeds[iter][i], iter_data_dir), callback=counter_callback)
                    result_objs.append(result)

                results = [result.get() for result in result_objs]
                all_results.append(results)
        
        results_array = np.array(results)
        largest_error_inds = np.argpartition(abs(results_array[:,2] - results_array[:,1]), k_largest)[-k_largest:]
        recheck_seeds = [list(zip(*results))[0][ind] for ind in largest_error_inds]  # index seeds in this way to prevent float conversion
        new_seeds = [seeding.create_seed() for i in range(k_largest)]
        seeds[iter+1] = recheck_seeds + new_seeds
        # print(len(results))

    # print(results)
    #%%
    # results = [result.get() for result in result_objs]
    results_array = np.array(results)
    print(np.mean(results_array[:,0]), np.std(results_array[:,0]))
    print(np.mean(results_array[:,1]), np.std(results_array[:,1]))
    print(np.mean(results_array[:,2]/60), np.std(results_array[:,2]/60))


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

