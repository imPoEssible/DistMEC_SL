"""
Utils.py

Update Date: 6/2/22

Summary:
helper functions to generate simulation testbed, and run rounds of reservation based distributed UCB
"""

import numpy as np
import itertools
from classes.solver import *
import pdb
# from classes.Server import *

def gen_eq_locs(space_1d, nums, offset = 0.5):
    # Generate well spread out locations in square space
    num_across = int(np.floor(np.sqrt(nums)))
    locs = []
    
    inc = space_1d/num_across
    
    for i,j in itertools.product(range(num_across), range(num_across)):
        locs += [(i*inc+offset, j*inc+offset)]
    
    return locs

def gen_rand_locs(space_1d, nums):
    locs = []
    for i in range(nums):
        locs += [(np.random.uniform(low=0.0, high=space_1d),np.random.uniform(low=0.0, high=space_1d))]
    return locs

def obtain_w(Users, num_users, num_svrs): # checked
    
    w_curr = np.zeros([num_users,num_svrs])
    for i in range(num_users):
        w_curr[i] = Users[i].reward_scale[Users[i].usr_place]
    
    return w_curr

def update_user_locs(Users): # Checked
    
    for i in range(len(Users)):
        Users[i].next_loc()
    return
    
def get_arms_list(Users): # Checked
    arms = []
    for i in range(len(Users)):
        arms+= [Users[i].choose_arm()]
    return arms

def sort_server_results(arms_list, Servers, Users):

    reserve_id_dict = {}
    reserve_max_val_dict = {}
    reserve_time_dict = {}
    reward_dict = {}
    collision_flag_dict = {}
    random_serve_dict = {}

    for s in range(len(Servers)):
        usr_idxs = np.argwhere(np.array(arms_list) == s).flatten()
        scales = np.zeros(usr_idxs.shape[0])
        w_est = np.zeros(usr_idxs.shape[0])
        stay_times = np.zeros(usr_idxs.shape[0])
        for u in range(usr_idxs.shape[0]):
            scales[u] = Users[usr_idxs[u]].reward_scale[Users[usr_idxs[u]].usr_place,s]
            w_est[u] =  Users[usr_idxs[u]].ucb_present[s]
            stay_times[u] = Users[usr_idxs[u]].expected_time

        user_list = usr_idxs.tolist()
        scales_list = scales.tolist()
        w_est_list = w_est.tolist()
        stay_times_list = stay_times.tolist()

        s_result = Servers[s].receive_users(user_list, scales_list, w_est_list, stay_times_list, len(Users))
        reserve_id, reserve_max_val, reserve_time  = s_result[0],s_result[1],s_result[2]
        reward, collision_flag, random_serve_idx = s_result[3],s_result[4], s_result[5]
        
        reserve_id_dict[s] = reserve_id
        reserve_max_val_dict[s] = reserve_max_val
        reserve_time_dict[s] = reserve_time
        reward_dict[s] = reward
        collision_flag_dict[s] = collision_flag
        random_serve_dict[s] = random_serve_idx
    
    return reserve_id_dict,reserve_max_val_dict ,reserve_time_dict ,reward_dict , collision_flag_dict, random_serve_dict


def update_user_info(Users, arms_list, reserve_id_dict,reserve_max_val_dict ,
                     reserve_time_dict ,reward_dict ,collision_flag_dict, random_serve_dict, reservation_mode = True):
    # update UCB information from user 
    for u in range(len(Users)):
        arm_id = arms_list[u]
        reward = reward_dict[arm_id][u]
        collision_flag = collision_flag_dict[arm_id]
        max_reward = reserve_max_val_dict[arm_id]
        wait_time = reserve_time_dict[arm_id]
        chosen_idx = reserve_id_dict[arm_id]
        random_served_idx = random_serve_dict[arm_id]
        Users[u].receive_reward(arm_id, reward, collision_flag, max_reward, wait_time, 
                                chosen_idx, random_served_idx, reservation_mode)
    return


def expected_reward_collision_sensing(arms, mus, w, data_mu = None, soft_collision = True):
    exp_mus = np.zeros(len(arms)) # arms - each element is user, and value is server they pull
    collision_counter = 0
    num_servers =  mus.shape[1]
    
    if data_mu is None:
        data_mu = np.ones(mus.shape[0])
        
    
    for i in range(len(arms)):
        
        if arms[i] >= 0 and arms[i] < num_servers:
        
            num_simul_pulls = np.argwhere(np.array(arms)==arms[i]).flatten().shape[0]
            if num_simul_pulls == 1:
                exp_mus[i] = w[i, arms[i]]* mus[i, arms[i]] * data_mu[i]
            else:            
                if soft_collision: # Here collision doesn't lead to loss of reward
                    # Randomly select one user to receive the reward and distribute it among all simultaneous pulls
                    reward = 1 / num_simul_pulls * w[i, arms[i]] * mus[i, arms[i]] * data_mu[i]
                    exp_mus[i] += reward                    
                collision_counter += 1
                
        
    return np.sum(exp_mus), collision_counter

def get_user_locs(Users):
    usr_loc_list = []
    for i in range(len(Users)):
        usr_loc_list += [Users[i].usr_place]
        
    return usr_loc_list

# Alter to 
def explore_rounds(Users, num_users, Servers, mu, regret, collision_count, 
                   optimal_reward = None, usr_move_flag = False, rounds=1,
                   skip_optimal = False, data_mu = None):

    arms = list(range(num_users)) 
    num_svrs = len(Servers)
    
    if data_mu is None:
        data_mu = np.ones(num_users)
    
    for j in range(rounds):
        for i in range(num_svrs):
            w = obtain_w(Users, num_users, num_svrs)
            
            if skip_optimal:
                temp_rwd = 10
                optimal = arms, temp_rwd
            else:
                optimal = offline_optimal_action(w, mu, data_mu)
            
            if optimal_reward is not None:
                optimal_reward[j*(num_svrs) + i] = optimal[1]
            
            reward_exp_now, collision_count[j*(num_svrs) + i] = expected_reward_collision_sensing(arms, mu, w, data_mu)
            regret[j*(num_svrs) + i] = optimal[1] - reward_exp_now

            svr_res = sort_server_results(arms, Servers, Users)
            update_user_info(Users, arms, svr_res[0], svr_res[1], svr_res[2], svr_res[3], svr_res[4], svr_res[5])
            if usr_move_flag:
                update_user_locs(Users)

            arms = sweep_init_next(arms, num_svrs)
    
    return
    
def explore_rounds2(Users, num_users, Servers, mu, regret, collision_count, 
                   optimal_reward=None, usr_move_flag=False, rounds=1,
                   skip_optimal=False, data_mu=None):

    num_arms = len(Servers)
    num_users_per_arm = min(num_users, num_arms)  # Number of users per arm (subset of users)

    if data_mu is None:
        data_mu = np.ones(num_users)

    for j in range(rounds):
        selected_users = np.random.choice(num_users, num_users_per_arm, replace=False)  # Randomly select users

        for i, server in enumerate(Servers):
            arms = selected_users[i * num_users_per_arm: (i + 1) * num_users_per_arm]  # Users for this arm

            w = obtain_w(Users, num_users, num_arms)

            if skip_optimal:
                temp_rwd = 10
                optimal = arms, temp_rwd
            else:
                optimal = offline_optimal_action(w, mu, data_mu)

            if optimal_reward is not None:
                optimal_reward[j * num_arms + i] = optimal[1]

            reward_exp_now, collision_count[j * num_arms + i] = expected_reward_collision_sensing(arms, mu, w, data_mu)
            regret[j * num_arms + i] = optimal[1] - reward_exp_now

            svr_res = sort_server_results(arms, Servers, Users)
            update_user_info(Users, arms, svr_res[0], svr_res[1], svr_res[2], svr_res[3], svr_res[4], svr_res[5])
            if usr_move_flag:
                update_user_locs(Users)

    return

    
def play_round(Users, Servers, mu, regret, collision_count, 
               usr_move_flag = False, reservation_mode = True, debugger = False, optimal = None, t = None,
               w = None, arms_override = None, data_mu = None, soft_collision = True):
    
    num_users = len(Users)
    num_svrs = len(Servers)
    
    if data_mu is None:
        data_mu = np.ones(num_users)
    
    if t is None:
        t = int(np.sum(Users[0].pulls))
    
    if type(w) is not np.ndarray:
        w = obtain_w(Users, num_users, num_svrs)
    
    if optimal == None:
        optimal = offline_optimal_action(w, mu, data_mu)
    
    if arms_override is not None:
        arms = arms_override
    else:
        arms = get_arms_list(Users)
    
    
    reward_exp_now, collision_count[t] = expected_reward_collision_sensing(arms, mu, w, data_mu, soft_collision)
    regret[t] = optimal[1] - reward_exp_now
    svr_res = sort_server_results(arms, Servers, Users)
    update_user_info(Users, arms, svr_res[0], svr_res[1], svr_res[2], svr_res[3], svr_res[4], 
                     svr_res[5], reservation_mode)
    if usr_move_flag:
        update_user_locs(Users)
            
    return svr_res

def copy_usr_loc(Users1, Users2):
    
    for u in range(len(Users1)):
        Users2[u].usr_place = Users1[u].usr_place
        Users2[u].expected_time_true = Users2[u].get_expected_time()
        
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Fix P
def fix_P(user):
    P = user.P
    
    for i in range(user.P.shape[1]):
        tot = np.sum(user.P[i])
        if tot != 1.0:
            user.P[i] = user.P[i]/tot
        
    return


def obtain_w_stationary(Users, num_users, num_svrs):
    
    w_curr = np.zeros([num_users,num_svrs])
    for i in range(num_users):
        w_curr[i] = Users[i].stationary_reward_scale
    
    return w_curr

def extract_centralized_case(Users, num_users, num_svrs):
    
    rewards_record = np.zeros([num_users,num_svrs])
    pulls_record = np.zeros([num_users,num_svrs])
    ucb = np.zeros([num_users,num_svrs])
    
    for i in range(num_users):
        rewards_record[i] = Users[i].param_summed
        pulls_record[i] = Users[i].pulls
        ucb[i] = Users[i].ucb_raw 
    
    return rewards_record, pulls_record, ucb