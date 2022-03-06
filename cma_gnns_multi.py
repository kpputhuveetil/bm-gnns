#%%
from re import S
from unittest.mock import NonCallableMagicMock
import gym, sys, argparse, multiprocessing, time, os, math
sys.path.insert(0, '/home/kpputhuveetil/git/vBM-GNNdev/assistive-gym-fem')
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
import glob
from assistive_gym.learn import make_env

#%%
def grasp_on_cloth(action, cloth_initial):
    scale = [0.44, 1.05]
    grasp_loc = action[0:2]*scale

    dist = []
    for i, v in enumerate(cloth_initial):
        v = np.array(v)
        d = np.linalg.norm(v[0:2] - grasp_loc)
        dist.append(d)
    # * if no points on the blanket are within 2.8 cm of the grasp location, exit 
    return (np.any(np.array(dist) < 0.028))

def cost_function(action, human_pose, cloth_initial, graph, model):
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
    cost, covered_status = get_cost(pred, human_pose, action)

    return [cost, pred, covered_status]


def get_cost(cloth_final_2D_pos, human_pose, action):
    #! IMPLEMENT CHECKS FOR POINTS INITIALLY UNCOVERED
    #! IMPLEMENT LOW REWARD FOR GRASPS OFF OF THE BLANKET
    all_possible_target_limbs = [
            [0], [0,1], [0,1,2], 
            [3], [3,4], [3,4,5],
            [6], [6,7], [6,7,8],
            [9], [9,10], [9,10,11],
            [6,7,8,9,10,11], [3,4,5,9,10,11]]
    # target_limb_code = 12
    target = all_possible_target_limbs[target_limb_code]
    # print(target)

    covered_status = []

    for joint_pos in human_pose.tolist():
        covered = False
        for point in cloth_final_2D_pos:
            if np.linalg.norm(point - joint_pos) <= 0.05:
                covered = True
                break
        if covered:
            covered_status.append(True)
        else:
            covered_status.append(False)
    # print(covered_status)
    head_ind = len(covered_status)-1
    target_uncovered_reward = 0
    nontarget_uncovered_penalty = 0
    head_covered_penalty = 0
    for ind, cov in enumerate(covered_status):
        if ind in target and cov is False:
            target_uncovered_reward += 1
        elif ind == head_ind and cov is True:
            head_covered_penalty = 1
        elif ind not in target and ind != head_ind and cov is False:
            nontarget_uncovered_penalty += 1
    # print(target_uncovered_reward, nontarget_uncovered_penalty, head_covered_penalty)
    target_uncovered_reward = 100*(target_uncovered_reward/len(target))
    nontarget_uncovered_penalty = -100*(nontarget_uncovered_penalty/len(target))
    head_covered_penalty = -200*head_covered_penalty

    scale = [0.44, 1.05]
    grasp_loc = action[0:2]*scale
    release_loc = action[2:4]*scale
    reward_distance_btw_grasp_release = -150 if np.linalg.norm(grasp_loc - release_loc) >= 1.5 else 0

    reward = target_uncovered_reward + nontarget_uncovered_penalty + head_covered_penalty + reward_distance_btw_grasp_release
    
    cost = -reward
    # print(reward, cost)

    return cost, covered_status

def generate_figure(cloth_initial, pred, human_pose, covered_status):
    
    initial_gt = np.array(cloth_initial)

    aspect = (4, 6)

    fig, ax1 = plt.subplots(figsize=aspect)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    s2 = ax1.scatter(pred[:,0], pred[:,1], alpha=0.6)
    s3 = ax1.scatter(human_pose[:,0], human_pose[:,1], c=covered_status, marker="X", cmap='Dark2', s=90)
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.8, 1.2])
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.invert_yaxis()

    # plt.show()

    fig.legend((s1,s2,s3), ('Initial GT', 'Final Predicted', 'Human'), 'lower center', ncol=3, borderaxespad=0.3)

    return fig

def generate_figure_sim_results(cloth_initial, pred, human_pose, covered_status, info):
    
    initial_gt = np.array(cloth_initial)

    # aspect = (4, 6)
    aspect = (12, 10)

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=aspect)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)

    s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    s2 = ax1.scatter(pred[:,0], pred[:,1], alpha=0.6)
    s3 = ax1.scatter(human_pose[:,0], human_pose[:,1], c=covered_status, marker="X", cmap='Dark2', s=90)
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.9, 1.05])
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.invert_yaxis()

    final_sim = np.array(info["cloth_final_subsample"])

    ax2.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
    # s3 = ax2.scatter(pred.detach()[:,0], pred.detach()[:,1], color='red', alpha=0.6)
    s4 = ax2.scatter(final_sim[:,0], final_sim[:,1],  color='green', alpha=0.6)
    ax2.scatter(human_pose[:,0], human_pose[:,1], c=info["covered_status_sim"], marker="X", cmap='Dark2', s=90)
    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.9, 1.05])
    ax2.set_xlabel('x position')
    ax2.set_ylabel('y position')
    ax2.invert_yaxis()
    # plt.show()

    # plt.show()

    fig.legend((s1,s2,s3,s4), ('Initial GT', 'Final Predicted', 'Human', 'Final Sim'), 'lower center', ncol=2, borderaxespad=0.3)

    return fig

def load_model(checkpoint):
    checkpoint_path = osp.join(checkpoint, 'model_249.pth')
    epochs = 300
    proc_layers = 8
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

def gnn_cma(env_name, i, model):
    # print(i)

    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    done = False
    #env.render()
    env.set_seed_val(seeding.create_seed())
    human_pose = env.reset()
    human_pose = np.reshape(human_pose, (-1,2))
    cloth_initial = env.get_cloth_state()

    # # cloth_initial, human_pose = env.reset()
    # # print(cloth_initial)
    # # f = '/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/data_2089/raw/c0_331897332481794079_pid15959.pkl'
    # raw_data = pickle.load(open(f, "rb"))
    # cloth_initial = raw_data['info']['cloth_initial'][1]
    # # human_pose = raw_data['observation'][0]
    # human_pose = np.reshape(raw_data['observation'][0], (-1,2))

    # print('captured cloth initial')
    graph = BM_Graph(
        voxel_size=0.05, 
        edge_threshold=0.06, 
        action_to_all=True, 
        cloth_initial=cloth_initial)
    # print('graph constructed')

    popsize = 8

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': popsize, 'maxfevals': 500, 'tolfun': 1e-2, 'tolflatfitness': 10, 'tolfunhist': 1e-20}) # , 'tolfun': 10, 'maxfevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)
    x0 = [0.5, 0.5, -0.5, -0.5]
    # x0 = np.random.uniform(-1,1,4)
    sigma0 = 0.1

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
        
        actions = es.ask()
        output = [cost_function(x, human_pose, cloth_initial, graph, model) for x in actions]
        t1 = time.time()
        output = [list(x) for x in zip(*output)]
        costs = output[0]
        preds = output[1]
        covered_status = output[2]
        # print(-1*np.array(costs))
        es.tell(actions, costs)
        
        if (best_cost is None) or (np.min(costs) < best_cost):
            best_cost = np.min(costs)
            best_cost_ind = np.argmin(costs)
            best_reward = -best_cost
            best_action = actions[best_cost_ind]
            best_pred = preds[best_cost_ind]
            best_covered_status = covered_status[best_cost_ind]
            best_time = t1 - t0
            best_fevals = fevals
            best_iterations = iterations
        if best_cost == 100:
            break
    observation, env_reward, done, info = env.step(best_action)
    print(info,keys)
    ## fig_dir = '/home/kpputhuveetil/git/vBM-GNNdev/cmaes_images'

    pid = os.getpid()
    fig_dir = '/home/kpputhuveetil/git/vBM-GNNdev/cmaes_eval_data/images'
    fig = generate_figure_sim_results(cloth_initial, best_pred, human_pose, best_covered_status, info)
    fig_name = f'tl{target_limb_code}_time={best_time:.2f}_rew={best_reward:.2f}_sim-rew={env_reward:.2f}_eval{i+counter}__{pid}.png'
    fig.savefig(osp.join(fig_dir, fig_name))
    print(fig_name, best_action)

    save_data_to_pickle(i, env, info, env_reward, best_action, human_pose, fig, best_reward, best_time, best_pred, best_covered_status, target_limb_code, best_fevals, best_iterations)



    del env

    return best_reward, env_reward

def save_data_to_pickle(i, env, info, env_reward, best_action, human_pose, fig, best_reward, best_time, best_pred, best_covered_status, target_limb_code, best_fevals, best_iterations):
    pid = os.getpid()
    filename = f"tl{target_limb_code}_c{i+counter}_{env.seed_val}_pid{pid}"
    pkl_loc = '/home/kpputhuveetil/git/vBM-GNNdev/cmaes_eval_data'
    with open(os.path.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "human_pose":human_pose, 
            "info":info, 
            "action":best_action,
            "env_reward":env_reward,
            "figure": fig,
            'best_reward':best_reward,
            'best_time':best_time,
            'best_pred_final':best_pred,
            'best_covered_status': best_covered_status,
            'target_limb_code':target_limb_code,
            'best_fevals':best_fevals,
            'best_iterations':best_iterations}, f)
    output = [i, filename, pid]
    del env
    return output

def counter_callback(output):
    global counter
    counter += 1
    print(f"{counter} - Trial Completed: CMA-ES Best Reward:{output[0]}, Sim Reward: {output[1]}")
    # print(f"{counter} - Trial Completed: {output[0]}, Worker: {output[2]}, Filename: {output[1]}")
    # print(f"Trial Completed: {output[0]}, Worker: {os.getpid()}, Filename: {output[1]}")


# %%
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    # parser.add_argument('--env', default='ScratchItchJaco-v1',
    #                     help='Environment to test (default: ScratchItchJaco-v1)')
    # args = parser.parse_args()
    env_name = "GNNDatasetCollect-v1"

    # current_dir = os.getcwd()
    # pkl_loc = os.path.join(current_dir,'gnn_test_dc/pickle544')
    # pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)

    # * make the enviornment, set the specified target limb code and an initial seed value
    checkpoint = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/test/checkpoints'
    model = load_model(checkpoint)

    num_proc = 8

    # data_dir = "/home/kpputhuveetil/git/vBM-GNNdev/bm-gnns/data_2089"
    # data_dir = osp.join(data_dir, 'raw/*.pkl')
    # filenames = glob.glob(data_dir)
    # eval_files = filenames[-100:]
    target_limb_code = 13

    counter = 0

    # reserve one cpu to keep working while collecting data
    num_processes = multiprocessing.cpu_count() - 1

    # num data points to collect
    trials = 1000

    trials = 30
    num_processes = 2
    all_results = []
    result_objs = []
    for j in range(math.ceil(trials/num_processes)):
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_processes):
                result = pool.apply_async(gnn_cma, args = (env_name, i, model), callback=counter_callback)
                result_objs.append(result)

            results = [result.get() for result in result_objs]
            all_results.append(results)
    print(len(results))
    
    print(results)

# %%
# all_results_92 = results
# all_results_4 = results
# %%
all_results = all_results_92 + results
# %%
res = all_results[0:100]
# %%
res=np.array(res)
# %%
np.mean(res[:,0])
#%%
np.std(res[:,0])
# %%
np.mean(res[:,1])
# %%
np.std(res[:,1])
# %%
