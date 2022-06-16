import numpy as np

class Server():
    def __init__(self, loc, mu, s_idx):
        self.loc = loc
        self.mu = mu
        self.s_idx = s_idx
       
    def receive_users(self, user_list, scales_list, w_est_list, stay_times_list, num_users):
        
        
        
        # if 1 pull
        if len(user_list) == 1:
            reserve_id = user_list[0]
            reward = np.zeros(num_users)
            reward[user_list[0]] = scales_list[0] *  np.random.binomial(n=1,p=self.mu[reserve_id, self.s_idx])
            reserve_max_val = scales_list[0]* w_est_list[0]
            reserve_time = stay_times_list[0]
            collision_flag = False
        elif len(user_list) > 1:
            collision_flag = True
            reward = np.zeros(num_users)
            reserve_max_val_list = np.zeros(len(user_list))
            for i in range(len(user_list)):
                reward[user_list[i]] = scales_list[i] *  np.random.binomial(n=1,p=self.mu[user_list[i], self.s_idx])
                reserve_max_val_list[i] = scales_list[i] * w_est_list[i]
            
            chosen_idx = np.random.choice(np.flatnonzero(reserve_max_val_list == reserve_max_val_list.max()))
#             print(chosen_idx)
            
            reserve_id = user_list[(chosen_idx)]
            reserve_max_val = reserve_max_val_list[(chosen_idx)]
            reserve_time = stay_times_list[(chosen_idx)]
            
                
        else: # no users pull this arm
            reserve_id, reserve_max_val, reserve_time, reward, collision_flag = None, None, None, None, False
            
        
        return reserve_id, reserve_max_val, reserve_time, reward, collision_flag