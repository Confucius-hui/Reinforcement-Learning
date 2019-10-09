import numpy as np
import pandas as pd
import time


np.random.seed(2019)  ##固定随机数

N_STATES = 6 # the length of the 1 dimensional world
ACTION = ['left','right'] #available actions
EPSTION = 0.9 #greedy police
ALPHA = 0.1 # learning rate
LAMBDA = 0.9 #discount factor
MAX_EPISODES = 13 #maximum episodes
FRESH_TIME = 0.3 #fresh time for one move

#build Q table
def build_q_table(n_states,actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions)
    #print(table)
    return table

#choose action
def choose_aciton(state,q_table):
    # now state
    state_actions = q_table.iloc[state,:]
    if state_actions.all() == 0 or np.random.uniform()>EPSTION:
        action_name = np.random.choice(ACTION)
    else:
        action_name = state_actions.argmax()
    return action_name

# cal value
def get_env_feedback(S,A):
    if A == 'right':
        if S == N_STATES-2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S #起始点
        else:
            S_ = S - 1
    return S_,R
def update_env(S,episode,step_counter):
    env_list = ['-']*(N_STATES-1)+['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_step = %s' %(episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r               ',end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def rf():
    q_table = build_q_table(N_STATES,ACTION)

    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S,episode,step_counter)
        while not is_terminated:
            A = choose_aciton(S,q_table)
            S_,R = get_env_feedback(S,A)
            q_predict = q_table.ix[S,A]
            if S_ != 'terminal': # not get target
                q_target = R + LAMBDA*q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminated = True #get target

            q_table.ix[S,A]+=ALPHA*(q_target-q_predict)
            S = S_

            update_env(S,episode,step_counter+1)
            step_counter+=1
if __name__ == '__main__':
    rf()