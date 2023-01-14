import gym, sys, argparse, multiprocessing, time, os
from gym.utils import seeding
import numpy as np
import cma
from cma.optimization_tools import EvalParallel2
import pickle
import pathlib
from cma_gnn_util import *
from assistive_gym.learn import make_env
# import assistive_gym


def cost_function(action):
    pid = os.getpid()
    t0 = time.time()

    observation = env.reset()
    done = False
    while not done:
        # env.render()
        observation, reward, done, info = env.step(action)
        t1 = time.time()
        cost = -reward
        elapsed_time = t1 - t0

    return [cost, observation, elapsed_time, pid, info]

# # TEST COST FUNCTION
# def cost_function(action):
#     pid = os.getpid()
#     t0 = time.time()

#     done = False
#     cost = 0
#     while not done:
#         t1 = time.time()
#         cost = (action[0] - 3) ** 2 + (10 * (action[1] + 2)) ** 2 + (10 * (action[2] + 2)) ** 2 + (10 * (action[3] - 3)) ** 2
#         observation = 0
#         elapsed_time = t1 - t0
#         done = True

#     return [cost, observation, elapsed_time, pid, {}]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='CMA-ES sim optimization')
    # parser.add_argument('--env', default='BodiesUncoveredGNN-v1', help='env', required=True)
    # # parser.add_argument('--target-limb-code', required=True,
    # #                         help='Code for target limb to uncover, see human.py for a list of available target codes')
    # # parser.add_argument('--run_id', help='id for the run (this code can be run 4 times simultaneously', required=True)
    # args, unknown = parser.parse_known_args()
    # current_dir = os.getcwd()
    # pkl_loc = os.path.join(current_dir,'cmaes_data_collect/pickle')
    # pstate_loc = os.path.join(current_dir,'cmaes_data_collect/bullet_state')
    
    # pathlib.Path(pstate_loc).mkdir(parents=True, exist_ok=True)


    # * make the enviornment, set the specified target limb code and an initial seed value
    env_name = "BodiesUncoveredGNN-v1"
    # env = gym.make("BodiesUncoveredGNN-v1")
    # env.set_seed_val(seeding.create_seed())
    # if args.target_limb_code == 'random':
    #     env.set_target_limb_code()
    # else:
    #     target_limb = int(args.target_limb_code)
    #     env.set_target_limb_code(target_limb)

    # * set the number of processes to 1/4 of the total number of cpus
    # *     collect data for 4 different target limbs simultaneously by running this script in 4 terminals
    # num_proc = multiprocessing.cpu_count()//4
    num_proc = 8
    max_fevals = 150

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': num_proc, 'maxfevals': max_fevals, 'tolfun': 1e-11, 'tolflatfitness': 20, 'tolfunhist': 1e-20}) # , 'tolfun': 10, 'maxfevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)
    
    sigma0 = 0.2
    reward_threshold = 95


    total_fevals = 0

    i = -1

    eval_loc = '/home/kpputhuveetil/git/vBM-GNNdev/trained_models/FINAL_MODELS/standard_2D_10k_epochs=250_batch=100_workers=4_1668718872/cma_evaluations/combo_var_150'
    pkl_loc = os.path.join(eval_loc, 'sim_dym_eval')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)

    env = make_env(env_name, coop=False)
    env.set_env_variations(
        collect_data = False,
        blanket_pose_var = True,
        high_pose_var = True,
        body_shape_var = True)
    
    
    # * repeat optimization for x number of human poses
    while i < 500:
        i += 1
        data_filename = os.path.join(eval_loc, f'tl_and_seeds_{i}.pkl')
        if os.path.exists(data_filename):
            f = open(data_filename,'rb')
            target_limb_code, seed = pickle.load(f)[i]
            f.close()
            os.rename(data_filename, os.path.join(eval_loc, f'tl_and_seeds_{i+1}.pkl'))
            # print(target_limb_code, seed)
            
            x0 = set_x0_for_cmaes(target_limb_code)
            # x0 = bounds/2.0
            env.set_target_limb_code(target_limb_code)
            env.set_seed_val(seed)
            print(x0)
        else:
            continue

        # env.set_pstate_file(os.path.join(pstate_loc, filename +".bullet"))


        print(f"Pose number: {i}, total fevals: {total_fevals}, target limb code: {env.target_limb_code}, enviornment seed: {env.seed_val}")
        fevals = 0
        iterations = 0
        t0 = time.time()

        # * initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        best_cost = None
        # * evaluate cost function in parallel over num_cpus/4 processes
        with EvalParallel2(cost_function, number_of_processes=num_proc) as eval_all:
            # * continue running optimization until termination criteria is reached
            while not es.stop():
                iterations += 1
                fevals += num_proc
                total_fevals += num_proc
                
                actions = es.ask()
                output = eval_all(actions)
                t1 = time.time()
                output = [list(x) for x in zip(*output)]
                costs = output[0]
                observations = output[1]
                elapsed_time = output[2]
                pids = output[3]
                info = output[4]
                es.tell(actions, costs)
                
                rewards = [-c for c in costs]
                mean = np.mean(rewards)
                min = np.min(rewards)
                max = np.max(rewards)
                total_elapsed_time = t1-t0

                #! TESTING ONLY
                # if iterations == 1: costs = [-95]
                # iterations = 300

                if (best_cost is None) or (np.min(costs) < best_cost):
                    best_cost = np.min(costs)
                    best_cost_ind = np.argmin(costs)
                    best_reward = -best_cost
                    best_action = actions[best_cost_ind]
                    best_time = total_elapsed_time
                    best_fevals = fevals
                    best_iterations = iterations
                    best_info = info[best_cost_ind]
                    best_pid = pids[best_cost_ind]
                if best_reward >= reward_threshold:
                    break

                print(f"Pose: {i}, iteration: {iterations}, total fevals: {total_fevals}, fevals: {fevals}, elapsed time: {total_elapsed_time:.2f}, mean reward = {mean:.2f}, min/max reward = {min:.2f}/{max:.2f}, best = {best_reward}")

                # * if any of the processes reached the reward_threshold, stop optimizing
                if np.any(np.array(costs) <= -reward_threshold):
                    print("Reward threshold reached")
                    break
                if fevals >= max_fevals:
                    print(f"No solution found after {max_fevals} fevals")
                    break

            es.result_pretty()

            # * open the pickle file to send optimization data to
            filename = f"tl{target_limb_code}_c{i}_{seed}_pid{best_pid}_sim_dyn.pkl"
            f = open(os.path.join(pkl_loc, filename),"wb")
            pickle.dump({
                    "seed": seed,
                    "target_limb": env.target_limb_code, 
                    "iteration": iterations,
                    "fevals": best_fevals, 
                    "total_elapsed_time":best_time, 
                    "actions": best_action,
                    "rewards": best_reward, 
                    "observations":observations, #? save only the first observation (they are all the same since pose is the same)
                    "info":best_info}, f)
            f.close()
            print("Data saved to file:", filename)
            print()

        # env.set_seed_val(seeding.create_seed())
        # if args.target_limb_code == 'random':
        #     env.set_target_limb_code()
        # else:
        #     env.set_target_limb_code(target_limb)
        # pose_count += 1
