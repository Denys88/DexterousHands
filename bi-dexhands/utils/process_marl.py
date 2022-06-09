# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def get_AgentIndex(config):
    agent_index = []
    # right hand
    agent_index.append(eval(config["env"]["handAgentIndex"]))
    # left hand
    agent_index.append(eval(config["env"]["handAgentIndex"]))

    return agent_index
    
def process_MultiAgentRL(args,env, config, model_dir=""):

    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs

    if args.algo in ["mappo", "happo", "hatrpo"]:
        # on policy marl
        from algorithms.marl.runner import Runner
        marl = Runner(vec_env=env,
                    config=config,
                    model_dir=model_dir
                    )
    elif args.algo == 'maddpg':
        # off policy marl
        from algorithms.marl.maddpg.runner import Runner
        marl = Runner(vec_env=env,
            config=config,
            model_dir=model_dir
            )
    elif args.algo == 'marlgppo':
        from utils.rlgames_utils import RLGWrapper
        from rl_games.torch_runner import Runner
        env = RLGWrapper(env)
        config['params']['config']['vec_env'] = env
        config['params']['config']['env_info'] = env.get_env_info()
        config['params']['config']['num_actors'] = args.num_envs
        config['params']['config']['name'] = args.experiment
        runner = Runner()
        runner.load(config)
        if args.test:
            runner.run({
                'train': False,
                'play': True,
                'checkpoint' : args.checkpoint
            })
        else:
            runner.run({
                'train': True,
            })
    return marl
