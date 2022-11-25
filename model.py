import torch.nn as nn
from torch.distributions import Categorical
from transformers import TransfoXLModel, TransfoXLConfig
import torch
import numpy as np
import clip


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden, out_dim, n_hidden=0):
        super(DiscreteActor, self).__init__()
        self.modlist = [nn.Linear(input_dim, hidden),
                        nn.LayerNorm(hidden, elementwise_affine=False),
                        nn.ReLU()]
        if n_hidden > 0:
            self.modlist.extend([nn.Linear(hidden, hidden),
                                 nn.LayerNorm(hidden, elementwise_affine=False),
                                 nn.ReLU()] * n_hidden)
        self.modlist.extend([nn.Linear(hidden, out_dim),
                            nn.Softmax(dim=-1)])
        self.actor = nn.Sequential(*self.modlist).apply(orthogonal_init)

    def forward(self, states, deterministic=False):
        probs = self.actor(states)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs).squeeze()
        else:
            action = dist.sample().squeeze()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions.squeeze())
        entropy = dist.entropy()
        return log_prob, entropy


class SmallImpalaCNN(nn.Module):
    def __init__(self, observation_shape, channel_scale=1, hidden_dim=256):
        super(SmallImpalaCNN, self).__init__()
        self.obs_size = observation_shape
        in_channels = 3
        kernel1 = 8 if self.obs_size[1] > 9 else 4
        kernel2 = 4 if self.obs_size[2] > 9 else 2
        stride1 = 4 if self.obs_size[1] > 9 else 2
        stride2 = 2 if self.obs_size[2] > 9 else 1
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16*channel_scale, kernel_size=kernel1, stride=stride1),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=16*channel_scale, out_channels=32*channel_scale, kernel_size=kernel2, stride=stride2),
                                    nn.ReLU())

        in_features = self._get_feature_size(self.obs_size)
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_dim)

        self.hidden_dim = hidden_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

    def _get_feature_size(self, shape):
        if shape[0] != 3:
            dummy_input = torch.zeros((shape[-1], *shape[:-1])).unsqueeze(0)
            print(dummy_input.shape)
        else:
            dummy_input = torch.zeros((shape[0], *shape[1:])).unsqueeze(0)
        x = self.block2(self.block1(dummy_input))
        return np.prod(x.shape[1:])


class FrozenHopfield(nn.Module):
    def __init__(self, hidden_dim, input_dim, embeddings, beta):
        super(FrozenHopfield, self).__init__()
        self.rand_obs_proj = torch.nn.Parameter(torch.normal(mean=0.0, std=1 / np.sqrt(hidden_dim), size=(hidden_dim, input_dim)), requires_grad=False)
        self.word_embs = embeddings
        self.beta = beta

    def forward(self, observations):
        observations = self._preprocess_obs(observations)
        observations = observations @ self.rand_obs_proj.T
        similarities = observations @ self.word_embs.T / (
                    observations.norm(dim=-1).unsqueeze(1) @ self.word_embs.norm(dim=-1).unsqueeze(0) + 1e-8)
        softm = torch.softmax(self.beta * similarities, dim=-1)
        state = softm @ self.word_embs
        return state

    def _preprocess_obs(self, obs):
        obs = obs.mean(1)
        obs = torch.stack([o.view(-1) for o in obs])
        return obs


class HELM(nn.Module):
    def __init__(self, action_space, input_dim, optimizer, learning_rate, epsilon=1e-8, mem_len=511, beta=1,
                 device='cuda'):
        super(HELM, self).__init__()
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len

        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103', config=config)
        n_tokens = self.model.word_emb.n_token
        word_embs = self.model.word_emb(torch.arange(n_tokens)).detach().to(device)
        hidden_dim = self.model.d_embed
        hopfield_input = np.prod(input_dim[1:])
        self.frozen_hopfield = FrozenHopfield(hidden_dim, hopfield_input, word_embs, beta=beta)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_encoder = SmallImpalaCNN(input_dim, channel_scale=4, hidden_dim=hidden_dim)
        self.out_dim = hidden_dim*2
        self.actor = DiscreteActor(self.out_dim, 128, action_space.n).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.out_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.yield_trainable_params(), lr=learning_rate,
                                                             eps=epsilon)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        self.memory = None

    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if 'model.' in n:
                continue
            else:
                yield p

    def forward(self, observations):
        bs, *_ = observations.shape
        obs_query = self.query_encoder(observations)
        vocab_encoding = self.frozen_hopfield.forward(observations)
        out = self.model(inputs_embeds=vocab_encoding.unsqueeze(1), output_hidden_states=True, mems=self.memory)
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        hiddens = out.last_hidden_state[:, -1, :].cpu().numpy()

        hidden = torch.cat([hidden, obs_query], dim=-1)

        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()

        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), hiddens

    def evaluate_actions(self, hidden_states, actions, observations):
        queries = self.query_encoder(observations)
        hidden = torch.cat([hidden_states, queries], dim=-1)

        log_prob, entropy = self.actor.evaluate(hidden, actions)
        value = self.critic(hidden).squeeze()

        return value, log_prob, entropy


class HELMv2(nn.Module):
    def __init__(self, action_space, input_dim, optimizer, learning_rate, epsilon=1e-8, mem_len=511, device='cuda'):
        super(HELMv2, self).__init__()
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len

        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103', config=config)
        n_tokens = self.model.word_emb.n_token
        word_embs = self.model.word_emb(torch.arange(n_tokens)).detach().to(device)
        self.we_std = word_embs.std(0)
        self.we_mean = word_embs.mean(0)
        self.vis_encoder = VisionBackbone()
        hidden_dim = self.model.d_embed

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_encoder = SmallImpalaCNN(input_dim, channel_scale=4, hidden_dim=hidden_dim)
        self.out_dim = hidden_dim*2
        self.actor = DiscreteActor(self.out_dim, 128, action_space.n).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.out_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.yield_trainable_params(), lr=learning_rate,
                                                             eps=epsilon)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        self.memory = None

    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if 'model.' in n or 'vis_encoder' in n:
                continue
            else:
                yield p

    def forward(self, observations):
        bs, *_ = observations.shape
        obs_query = self.query_encoder(observations)
        observations = self.vis_encoder(observations)
        observations = (observations - observations.mean(0)) / (observations.std(0) + 1e-8)
        observations = observations * self.we_std + self.we_mean
        out = self.model(inputs_embeds=observations.unsqueeze(1), output_hidden_states=True, mems=self.memory)
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        hiddens = out.last_hidden_state[:, -1, :].cpu().numpy()

        hidden = torch.cat([hidden, obs_query], dim=-1)

        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()

        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), hiddens

    def evaluate_actions(self, hidden_states, actions, observations):
        queries = self.query_encoder(observations)
        hidden = torch.cat([hidden_states, queries], dim=-1)

        log_prob, entropy = self.actor.evaluate(hidden, actions)
        value = self.critic(hidden).squeeze()

        return value, log_prob, entropy


class VisionBackbone(nn.Module):
    def __init__(self):
        super(VisionBackbone, self).__init__()
        print(f"Allocating CLIP...")
        self.model, preprocess = clip.load("RN50")
        self.transforms = preprocess
        preprocess.transforms = [preprocess.transforms[0], preprocess.transforms[1], preprocess.transforms[-1]]
        self.transforms = preprocess

        self.model.eval()
        self._deactivate_grad()

    def forward(self, observations):
        observations = self._preprocess(observations)
        out = self.model.encode_image(observations).float()
        return out

    def _deactivate_grad(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _preprocess(self, observation):
        return self.transforms(observation)


class MarkovianImpalaCNN(nn.Module):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate):
        super(MarkovianImpalaCNN, self).__init__()
        self.encoder = SmallImpalaCNN(obs_dim, channel_scale=4, hidden_dim=1024)
        hidden_dim = self.encoder.hidden_dim

        self.actor = DiscreteActor(hidden_dim, hidden=128, out_dim=action_dim).apply(orthogonal_init)

        critic_modules = []
        critic_modules.extend([nn.Linear(hidden_dim, 512),
                               nn.LayerNorm(512, elementwise_affine=False),
                               nn.ReLU()])
        critic_modules.append(nn.Linear(512, 1))
        self.critic = nn.Sequential(*critic_modules).apply(orthogonal_init)

        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=learning_rate)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")

    def forward(self, states):
        encoded = self.encoder(states)
        action, log_prob = self.actor(encoded)
        value = self.critic(encoded)
        return action.cpu().detach().numpy(), value.cpu().detach().squeeze().numpy(), log_prob.cpu().detach().numpy()

    def evaluate_actions(self, states, actions):
        encoded = self.encoder(states)
        log_probs, entropy = self.actor.evaluate(encoded, actions)
        values = self.critic(encoded).squeeze()
        return values, log_probs, entropy


class LSTMImpalaAgent(nn.Module):
    def __init__(self, action_dim, input_dim, optimizer, learning_rate, channel_scale=1, hidden_dim=256):
        super(LSTMImpalaAgent, self).__init__()
        self.encoder = SmallImpalaCNN(input_dim, channel_scale=channel_scale, hidden_dim=hidden_dim)
        self.hidden_dim = self.encoder.hidden_dim
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.actor = DiscreteActor(self.hidden_dim, 128, action_dim).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.hidden_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        self.hidden = None
        self.cell = None
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=learning_rate)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")

    def reset_states(self):
        self._init_hidden(1)

    def _init_hidden(self, batch_size):
        device = next(self.parameters()).device
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        self.cell = torch.zeros(1, batch_size, self.hidden_dim).to(device)

    def forward(self, state):
        bs, *_ = state.shape
        encoded = self.encoder(state)
        if self.hidden is None and self.cell is None:
            self._init_hidden(bs)
        last_hidden = np.array([self.hidden.cpu().numpy().squeeze(), self.cell.cpu().numpy().squeeze()])
        hidden, (self.hidden, self.cell) = self.lstm(encoded.unsqueeze(1), (self.hidden, self.cell))
        hidden = hidden[:, -1, :]
        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()
        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), last_hidden

    def evaluate_actions(self, states, actions, internals, detach_value_grad=False):
        bs, seqlen, *_ = states.shape
        states = states.reshape(bs*seqlen, *states.shape[2:])
        encoded = self.encoder(states)
        encoded = encoded.view(bs, seqlen, -1)
        internals = (internals[:, 0, 0, :].unsqueeze(0).contiguous(), internals[:, 0, 1, :].unsqueeze(0).contiguous())
        hidden, _ = self.lstm(encoded, internals)
        log_prob, entropy = self.actor.evaluate(hidden, actions)
        if detach_value_grad:
            hidden = hidden.detach()
        value = self.critic(hidden)
        return value, log_prob, entropy


def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def xavier_uniform_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.)
    return module
