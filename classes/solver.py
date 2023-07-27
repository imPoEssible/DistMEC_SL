import numpy as np
import pulp as lp
import itertools
import copy

# Offline optimal to calculate regret
def offline_optimal_action(W, mu, data_mu = None):
    
    action = np.ones_like(mu[:,0]) * -1
    
    C = np.append(mu,np.zeros([mu.shape[0],1]),axis=1)
    
    num_user = mu[:,0].shape[0]
    num_server = mu[0,:].shape[0]
    
    if data_mu is None:
        data_mu = np.ones(num_user)
    
    
    prob = lp.LpProblem("prob",lp.LpMaximize)
    
    # Import pulp and make decision variables
    dec_var = []
    for u,s in itertools.product(range(num_user),range(num_server+1)):
        dec_var += [(u,s-1)]

    X = lp.LpVariable.dicts("x", dec_var,lowBound=0,upBound = 1, cat='Integer')

    # Make Constraints - 1 pull per user
    usr_constraint = []
    for u in range(num_user):
        usr_constraint += [0]

    for u,s in itertools.product(range(num_user),range(num_server+1)):
        usr_constraint[u] += X[(u,s-1)]

    for u in range(num_user):
        prob += (usr_constraint[u] == 1)

    # Make constraint - 1 user per server
    svr_constraint = []

    for s in range(num_server):
        svr_constraint += [0]

    for u,s in itertools.product(range(num_user),range(num_server)):
        svr_constraint[s] += X[(u,s)]

    for s in range(num_server):
        prob += (svr_constraint[s] <= 1)

    # Make objective function
    obj = 0
    for u,s in itertools.product(range(num_user),range(num_server+1)):
        obj += W[u,s-1] * C[u,s-1] * X[(u,s-1)] * data_mu[u]
    prob += obj

    status = prob.solve(lp.PULP_CBC_CMD(msg=0))
    
    X2 = copy.deepcopy(X)

    for key in X2.keys():
        if X2[key].value() > 0:
            u,s = key
            action[u] = s
    
    rwd = 0
    
    for u in range(num_user):
        rwd += W[u, int(action[u])]* C[u,int(action[u])] * data_mu[u]
        
    new_action = []
    for i in range(action.shape[0]):
        new_action += [int(action[i])]
        
    del prob, X
    
    return new_action, rwd

def pull_super_arm(arms, mus, rewards_record, pulls_record):
    """
    arms - u x 1, vector, ith-element is user i, indicates which arm they will pull
    return - rewards of each arm drawn from bernoulli distribution 
    """
    
    for i in range(len(arms)): # looping through clients
        n = 1
        p = mus[i,arms[i]]
        s = np.random.binomial(n, p, 1)
        rewards_record[i,arms[i]] += [s]
        pulls_record[i,arms[i]] += 1
    return rewards_record, pulls_record
    
def update_ucb(reward_record, pulls_record, ucb, T, L=1):
    """
    Update decision variables for next round
    """
    
    num_users, num_svrs = reward_record.shape[0], reward_record.shape[1]
    for u in range(num_users):
        for s in range(num_svrs):
            if pulls_record[u,s] > 0:
                mean = reward_record[u,s]/pulls_record[u,s]
            else:
                mean = 0
            
            cb = np.sqrt(L * np.log(T)/ pulls_record[u,s])
            
            ucb[u,s] = mean + cb
    
    return ucb

    
def sweep_init_next(lst, num_svrs):
    for i in range(len(lst)):
        lst[i] += 1
        lst[i] = lst[i]%(num_svrs)
    
    return lst

def expected_reward(arms, mus, w, data_mu = None):
    
    num_user = mus[:,0].shape[0]
    num_server = mus[0,:].shape[0]
    
    if data_mu is None:
        data_mu = np.ones(num_user)
    
    exp_mus = np.zeros(len(arms))
    for i in range(len(arms)):
        if arms[i] >= 0:
            exp_mus[i] = w[i, arms[i]]* mus[i, arms[i]] * data_mu[i]
        
    return np.sum(exp_mus)