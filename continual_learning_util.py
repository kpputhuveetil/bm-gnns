import pickle, os
import torch
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def save_data_to_pickle(idx, seed, action, human_pose, target_limb_code, sim_info, cma_info, iter_data_dir, covered_status):
    if isinstance(covered_status, int) and covered_status == -1:
        return
    pid = os.getpid()
    filename = f"tl{target_limb_code}_c{idx}_{seed}_pid{pid}"

    raw_dir = osp.join(iter_data_dir, 'raw')
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    pkl_loc = raw_dir

    with open(os.path.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "action":action,
            "human_pose":human_pose, # [] necessary for correct unpacking by BMDataset
            'target_limb_code':target_limb_code,
            'sim_info':sim_info,
            'cma_info':cma_info,
            'observation':[sim_info['observation']], # exposed here for bm_dataset use later
            'info':sim_info['info']}, f)

def save_dataset(idx, graph, data, sim_info, action, human_pose, covered_status):
    # ! function behavior is not correct at the moment
    if isinstance(covered_status, int) and covered_status == -1:
        return


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