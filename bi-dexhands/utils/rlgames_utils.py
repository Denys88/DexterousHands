# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.common import vecenv

import gym
import numpy as np


class RLGWrapper(vecenv.IVecEnv):
    def __init__(self, env):
        self.env = env
        if self.env.num_agents == 1:
            self.observation_space = self.env.observation_space
            action_high = np.ones(self.env.action_space.shape[0])
            self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)
            self.state_space = None
        else:
            action_high = np.ones(self.env.action_space[0].shape[0])
            self.observation_space = self.env.observation_space[0]
            self.state_space = self.env.share_observation_space[0]
            self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)            
            
    def step(self, action):
        if self.env.num_agents == 1:
            res = self.env.step(action)
            obs, reward, is_done, info = res
            obs_dict = {
                'obs' : obs
            }
        else:
            action = action.reshape(-1, self.get_number_of_agents(), action.size()[1])
            res = self.env.step([action[:,0,:],action[:,1,:]])
            obs, state, reward, is_done, info, _ = res
            obs_dict = {
                'obs' : obs.reshape(-1,obs.size()[-1]),
                'states' : state[:,0,:]
            }        
        return obs_dict, reward.reshape(-1), is_done.reshape(-1), {}

    def reset(self):
        # todo add random init like in collab examples?
        res = self.env.reset()
        if self.env.num_agents == 1:
            obs = res
            obs_dict = {
                'obs' : obs
            }
        else:
            obs, state,_ = res
            obs_dict = {
                'obs' : obs.reshape(-1,obs.size()[-1]),
                'states' : state[:,0,:]
            }  
        return obs_dict

    def get_number_of_agents(self):
        return self.env.num_agents

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        info['state_space'] = self.state_space
        info['use_global_observations'] = True
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        return info




