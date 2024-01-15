from argparse import ArgumentParser
import glob
import json
import torch
import inspect
import os
import numpy as np
from utils import make_dmlab_env, make_maze_env, make_minigrid_env, make_procgen_env, make_miniworld_env
import random
from model import SHELM
np.random.seed(101)
torch.cuda.manual_seed(101)
random.seed(101)

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--random', action='store_true', help="Use random policy")
    parser.add_argument('--n_episodes', type=int, default=10, help="How many episodes to sample")
    return parser.parse_args()


def main():
    options = create_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model parameters and seed to resume training or perform transfer
    ckpt_files = glob.glob(os.path.join(options.ckpt_path, '**'), recursive=True)
    config_file = [f for f in ckpt_files if '.json' in f][0]
    with open(config_file, 'r') as infile:
        ckpt_config = json.load(infile)
        ckpt_config['n_envs'] = 1

    ckpt_file = [f for f in ckpt_files if '.pt' in f][0]
    ckpt = torch.load(ckpt_file)

    # create training environments
    if 'RandomMaze' in ckpt_config['env_id']:
        env = make_maze_env()()
    elif 'MiniGrid' in ckpt_config['env_id']:
        env = make_minigrid_env(ckpt_config['env_id'])()
    elif 'MiniWorld' in ckpt_config['env_id']:
        env = make_miniworld_env(ckpt_config['env_id'])()
    elif 'psychlab' in ckpt_config['env_id']:
        env = make_dmlab_env(ckpt_config['env_id'])()
    else:
        # create procgen environment
        env = make_procgen_env(id=ckpt_config['env_id'], num_envs=ckpt_config['n_envs'], num_levels=0)

    os.makedirs(os.path.join('episodes', ckpt_config['env_id']), exist_ok=True)

    model_args = inspect.getfullargspec(SHELM).args[3:]
    action_space = env.action_space
    input_dim = env.observation_space.shape
    arguments = [action_space, input_dim]
    for arg in model_args:
        arguments.append(ckpt_config[arg])

    print("Loading Model ...")
    agent = SHELM(*arguments).to(device)
    agent.eval()

    for attr, sd in ckpt['network'].items():
        if attr != 'seed':
            getattr(agent, attr).load_state_dict(sd)
    print("State dict loaded successfully...")

    saved = 0
    n_steps = []
    agent.memory = None
    while saved < options.n_episodes:
        obs = env.reset()
        done = False
        observations = [obs]
        memory = None
        agent.memory = memory
        ep_info = []
        steps = 0
        ret = 0
        while not done:
            with torch.no_grad():
                observation = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0).to(device)
                observation /= 255.
                encoded = agent.vis_encoder(observation)
                _, decoded = agent.get_top_k_toks(encoded, agent.clip_embs, k=1)

            if not options.random:
                with torch.no_grad():
                    action, values, log_prob, _ = agent.forward(observation)
            else:
                action = env.action_space.sample()

            obs, rew, done, infos = env.step(action)
            observations.append(obs)
            ret += rew
            ep_info.append((obs, decoded[0], action, rew))

            steps += 1

        obs_query = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0).to(device)
        obs_query /= 255.
        encoded = agent.vis_encoder(obs_query)
        _, decoded = agent.get_top_k_toks(encoded, agent.clip_embs, k=1)
        ep_info.append((obs, decoded[0], action, rew))

        print(f"Number of steps required: {steps}")
        n_steps.append(steps)
        print(f"Return: {ret}")

        saved += 1
        name = "agent" if not options.random else "random"
        np.save(f'episodes/{ckpt_config["env_id"]}_{name}_ep_info_{saved}', ep_info)


if __name__ == '__main__':
    main()
