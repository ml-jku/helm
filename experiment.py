from utils import *
from trainers.helm_trainer import HELMPPO
from trainers.lstm_trainer import LSTMPPO
from trainers.cnn_trainer import CNNPPO
import torch
import os
import uuid
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, DummyVecEnv


class Experiment:
    def __init__(self, config, experiment_id=None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if experiment_id is None:
            experiment_id = self._create_exp_id()
        else:
            experiment_id = experiment_id

        self.outpath = os.path.join('./experiments', config['model'], config['env'], experiment_id)
        os.makedirs(self.outpath, exist_ok=True)

    def _create_exp_id(self):
        return str(uuid.uuid4())

    def run(self, seed=None):

        # create training environments
        if 'RandomMaze' in self.config['env']:
            env = DummyVecEnv([make_maze_env() for _ in range(self.config['n_envs'])])
            env = VecMonitor(env)
        elif 'MiniGrid' in self.config['env']:
            env = DummyVecEnv([make_minigrid_env(self.config['env']) for _ in range(self.config['n_envs'])])
            env = VecNormalize(VecMonitor(env), norm_reward=True, norm_obs=False, clip_reward=1.)
        else:
            # create procgen environment
            env = make_procgen_env(id=self.config['env'], num_envs=self.config['n_envs'], num_levels=0)

        assert self.config['model'] in ['HELM', 'Impala-PPO', 'CNN-PPO'], \
            f"Model type {self.config['model']} not recognized!"

        if self.config['model'] == 'HELM':
            trainer = HELMPPO
        elif self.config['model'] == 'Impala-PPO':
            trainer = LSTMPPO
        else:
            trainer = CNNPPO

        model = trainer("MlpPolicy", env, verbose=1, tensorboard_log=f"{self.outpath}",
                        lr_decay=self.config['lr_decay'], ent_coef=self.config['ent_coef'],
                        ent_decay=self.config['ent_decay'], learning_rate=self.config['learning_rate'],
                        vf_coef=self.config['vf_coef'], n_epochs=int(self.config['n_epochs']),
                        ent_decay_factor=self.config['ent_decay_factor'], clip_range=self.config['clip_range'],
                        gamma=self.config['gamma'], gae_lambda=self.config['gae_lambda'],
                        n_steps=int(self.config['n_rollout_steps']), n_envs=int(self.config['n_envs']),
                        min_lr=self.config['min_lr'], min_ent_coef=self.config['min_ent_coef'],
                        start_fraction=self.config['start_fraction'], end_fraction=self.config['end_fraction'],
                        device=self.device, clip_decay=self.config['clip_decay'], config=self.config,
                        clip_range_vf=self.config['clip_range_vf'], seed=seed,
                        max_grad_norm=self.config['max_grad_norm'],
                        adv_norm=self.config.get('adv_norm', False),
                        save_ckpt=self.config.get('save_ckpt', False))

        model.learn(total_timesteps=self.config['n_steps'], eval_log_path=self.outpath)

        env.close()
