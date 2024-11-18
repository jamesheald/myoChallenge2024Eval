import os
import pickle
import time
import sys

sys.path.append('../')
sys.path.append('.')
sys.path.append("../utils")
sys.path.append('utils')

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym

from utils import RemoteConnection

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import mujoco
from myosuite.utils import gym
from myochallenge24_sac import SAC as final_policy
# from stable_baselines3 import SAC

"""
Define your custom observation keys here
"""
custom_obs_keys = ['myohand_qpos','myohand_qvel','act','pros_hand_qpos','pros_hand_qvel','start_pos','goal_pos','object_qpos','object_qvel','touching_body','time']

def load_policy(model_id):

    path = '/'.join(os.path.realpath('MPL_only').split('/')[:-1])
    print(path)
    MjModel = mujoco.MjModel.from_binary_path(os.path.join(path,'MPL_only.mjb'))
    print('MANIPULATION agent: mujoco model loaded')

    path = '/'.join(os.path.realpath('vec_stats_' + model_id).split('/')[:-1])
    print(path)
    vec_stats = os.path.join(path,'vec_stats_' + model_id + '.pkl')
    print('MANIPULATION agent: vec stats loaded')

    path = '/'.join(os.path.realpath('params_' + model_id).split('/')[:-1])
    print(path)
    learned_params = pickle.load(open(os.path.join(path,'params_' + model_id + '.pkl'),"rb"))
    print('MANIPULATION agent: params loaded')

    env_id = 'myoChallengeBimanual-v0'

    ordered_custom_obs_keys = ['myohand_qpos','myohand_qvel','act','pros_hand_qpos','pros_hand_qvel','start_pos','goal_pos','object_qpos','object_qvel','touching_body','time']
    control_variables = [215, 216, 17, 15, 16, 14, 10, 11, 13]
    task_variables = [191, 192, 193, 194, 195, 196]
    obs_variables = list(range(215)) ## TO USE THIS NEED TO UNCOMMENT CODE IN my_policies

    def make_env(env_id, seed, run, normalize_act, control_variables, obs_variables, ordered_custom_obs_keys, max_episode_steps, reward_id, max_num_mode_switches, wrap):
        def _init():
            env = gym.make(env_id, normalize_act=normalize_act, obs_keys=ordered_custom_obs_keys, max_episode_steps=max_episode_steps)
            if wrap:
                env = myoChallengeWrapper(env, control_variables, obs_variables, reward_id, max_num_mode_switches)
            env.unwrapped.seed(seed)
            return env
        set_random_seed(run)
        return _init

    run = 0
    num_cpu = 1
    reward_id = 1
    max_num_mode_switches = 'one' # 'one' (switch policy mode once only) or unbounded' (switch policy mode any number of times) 
    normalize_act = False
    max_episode_steps = 1000
    reward_type='dense'

    env_fn = make_env(env_id, run, run, normalize_act, control_variables, obs_variables, ordered_custom_obs_keys, max_episode_steps, reward_id, max_num_mode_switches, False) 

    model = final_policy("MlpPolicy", 
                          MjModel,
                          max_num_mode_switches,
                          vec_stats,
                          env=DummyVecEnv([env_fn]), 
                          seed=0, 
                          control_variables=control_variables,
                          task_variables=task_variables, 
                          obs_variables=obs_variables,
                          std_max=0.25, 
                          dynamics_dropout=0.1,
                          cocontraction=None,
                          num_MC_samples=1,
                          explore_coeff=10.,
                          source_mu=[-0.4 for _ in range(63)] + [0 for _ in range(17)],
                          buffer_size=int(1))
    
    model.policy.dynamics_state = model.policy.dynamics_state.replace(params=learned_params['dynamics'])
    model.policy.actor_state = model.policy.actor_state.replace(params=learned_params['actor'])
    model.policy.qf_state = model.policy.qf_state.replace(params=learned_params['qf'])

    return model

def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

def get_custom_observation(rc, obs_keys):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)

time.sleep(10)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

# compute correct observation space using the custom keys
shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

custom_environment_varibles = {'obs_keys':custom_obs_keys, 'normalize_act':False}
rc.set_environment_keys(custom_environment_varibles)

# path = '/'.join(os.path.realpath('MyMyoChallengePolicy').split('/')[:-1])
# print(path)
# # model = SAC.load(os.path.join(path,'MyMyoChallengePolicy'))
# model = final_policy.load(os.path.join(path,'MyMyoChallengePolicy'))
# print('MANIPULATION agent: policy loaded')

model_id = 'james_1'
model = load_policy(model_id)

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"MANI-MPL: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = rc.reset()

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    while not flag_trial:

        ################################################
        ## Replace with your trained policy.
        obs = get_custom_observation(rc, custom_obs_keys)
        action, __ = model.predict(obs, deterministic=True)
        ################################################

        base = rc.act_on_environment(action)

        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
