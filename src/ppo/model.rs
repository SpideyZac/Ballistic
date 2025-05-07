use burn::{
    module::{Param, ParamId},
    prelude::*,
};

use crate::{
    ppo::{
        config::PPOTrainingConfig, distribution::NormalDistribution,
        rollout_buffer::RolloutBufferBatch,
    },
    utils::tensor_std,
};

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
}

impl<B: Backend> Critic<B> {
    pub fn new(obs_dim: usize) -> Self {
        let device = &B::Device::default();
        let linear1 = nn::LinearConfig::new(obs_dim, 64).init(device);
        let linear2 = nn::LinearConfig::new(64, 64).init(device);
        let linear3 = nn::LinearConfig::new(64, 1)
            .with_initializer(nn::Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        Self {
            linear1,
            linear2,
            linear3,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x).tanh();
        let x = self.linear2.forward(x).tanh();
        let x = self.linear3.forward(x);
        x
    }
}

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    linear3: nn::Linear<B>,
    pub log_std: Param<Tensor<B, 2>>,
}

impl<B: Backend> Actor<B> {
    pub fn new(obs_dim: usize, action_dim: usize) -> Self {
        let device = Default::default();
        let linear1 = nn::LinearConfig::new(obs_dim, 64).init(&device);
        let linear2 = nn::LinearConfig::new(64, 64).init(&device);
        let linear3 = nn::LinearConfig::new(64, action_dim)
            .with_initializer(nn::Initializer::XavierUniform { gain: 0.01 })
            .init(&device);
        let log_std = Param::uninitialized(
            ParamId::new(),
            move |device, require_grad| {
                let mut tensor = Tensor::zeros([1, action_dim], device);
                if require_grad {
                    tensor = tensor.require_grad();
                }
                tensor
            },
            device,
            true,
        );

        Self {
            linear1,
            linear2,
            linear3,
            log_std,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x).tanh();
        let x = self.linear2.forward(x).tanh();
        let x = self.linear3.forward(x);
        x
    }
}

#[derive(Module, Debug)]
pub struct Agent<B: Backend> {
    actor: Actor<B>,
    critic: Critic<B>,
}

impl<B: Backend> Agent<B> {
    pub fn new(obs_dim: usize, action_dim: usize) -> Self {
        let actor = Actor::new(obs_dim, action_dim);
        let critic = Critic::new(obs_dim);

        Self { actor, critic }
    }

    pub fn get_value(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.critic.forward(x)
    }

    pub fn get_action_and_value(
        &self,
        x: Tensor<B, 2>,
        mut action: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 2>) {
        let device = &B::Device::default();

        let action_mean = self.actor.forward(x.clone());
        let action_logstd = self.actor.log_std.val().expand(action_mean.shape());
        let action_std = action_logstd.exp();

        let dist = NormalDistribution::new(&action_mean, &action_std, device);
        let sampled_action = dist.sample();
        let log_prob = dist.log_prob(&sampled_action);
        let entropy = dist.entropy();

        if let None = action {
            action = Some(sampled_action.clone());
        }

        let value = self.critic.forward(x);

        (
            action.unwrap(),
            log_prob.sum_dim(1).squeeze(1),
            entropy.sum_dim(1).squeeze(1),
            value,
        )
    }
}

pub struct TrainingBatch<B: Backend> {
    pub obs: Tensor<B, 2>,
    pub actions: Tensor<B, 2>,
    pub log_probs: Tensor<B, 1>,
    pub advantages: Tensor<B, 1>,
    pub returns: Tensor<B, 1>,
    pub values: Tensor<B, 1>,
}

fn generate_train_batch<B: Backend>(
    rollout_batch: &RolloutBufferBatch<B>,
    config: &PPOTrainingConfig,
) -> TrainingBatch<B> {
    let obs = rollout_batch.obs.clone();
    let actions = rollout_batch.actions.clone();
    let log_probs = rollout_batch.log_probs.clone();
    let values = rollout_batch.values.clone();
    let rewards = rollout_batch.rewards.clone();
    let not_dones = rollout_batch.not_dones.clone();

    let (returns, mut advantages) = get_gae(
        values.clone(),
        rewards,
        not_dones,
        config.gamma,
        config.gae_lambda,
    );

    if config.norm_adv {
        let advantages_shape = advantages.shape();
        let advantages_flat = advantages.flatten(0, 1).clone();
        let advantages_flat = (advantages_flat.clone() - advantages_flat.clone().mean())
            / (tensor_std(&advantages_flat) + 1e-8);
        advantages = advantages_flat.reshape(advantages_shape);
    }

    TrainingBatch {
        obs: obs.reshape([-1, config.obs_dim as i32]),
        actions: actions.reshape([-1, config.action_dim as i32]),
        log_probs: log_probs.flatten(0, 1),
        advantages: advantages.flatten(0, 1),
        returns: returns.flatten(0, 1),
        values: values.flatten(0, 1),
    }
}

fn get_gae<B: Backend>(
    values: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    not_dones: Tensor<B, 2>,
    gamma: f32,
    lambda: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut returns = vec![0.0 as f32; rewards.shape().num_elements()];
    let mut advantages = returns.clone();

    let mut running_return: f32 = 0.0;
    let mut running_advantage: f32 = 0.0;

    let rewards_data = rewards.to_data();
    let rewards_slice: &[f32] = rewards_data.as_slice().unwrap();
    let not_dones_data = not_dones.to_data();
    let not_dones_slice: &[f32] = not_dones_data.as_slice().unwrap();
    let values_data = values.to_data();
    let values_slice: &[f32] = values_data.as_slice().unwrap();

    for i in (0..rewards.shape().num_elements()).rev() {
        let reward = rewards_slice[i];
        let not_done = not_dones_slice[i];

        running_return = reward + gamma * running_return * not_done;
        running_advantage = reward - values_slice[i]
            + gamma
                * not_done
                * (*values_slice.get(i + 1).unwrap_or(&0.0) + lambda * running_advantage);

        returns[i] = running_return;
        advantages[i] = running_advantage;
    }

    (
        Tensor::<B, 1>::from_floats(returns.as_slice(), &Default::default())
            .reshape([returns.len(), 1]),
        Tensor::<B, 1>::from_floats(advantages.as_slice(), &Default::default())
            .reshape([advantages.len(), 1]),
    )
}
