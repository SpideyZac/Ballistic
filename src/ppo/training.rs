use burn::prelude::*;

use crate::{
    ppo::{config::PPOTrainingConfig, rollout_buffer::RolloutBufferBatch},
    utils::tensor_std,
};

pub fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub struct TrainingBatchTensors<B: Backend> {
    pub obs: Tensor<B, 2>,
    pub actions: Tensor<B, 2>,
    pub log_probs: Tensor<B, 1>,
    pub advantages: Tensor<B, 1>,
    pub returns: Tensor<B, 1>,
    pub values: Tensor<B, 1>,
}

pub fn generate_train_batch<B: Backend>(
    rollout_batch: &RolloutBufferBatch<B>,
    config: &PPOTrainingConfig,
) -> TrainingBatchTensors<B> {
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
            / (tensor_std(advantages_flat) + 1e-8);
        advantages = advantages_flat.reshape(advantages_shape);
    }

    TrainingBatchTensors {
        obs: obs.reshape([-1, config.obs_dim as i32]),
        actions: actions.reshape([-1, config.action_dim as i32]),
        log_probs: log_probs.flatten(0, 1),
        advantages: advantages.flatten(0, 1),
        returns: returns.flatten(0, 1),
        values: values.flatten(0, 1),
    }
}

pub fn get_gae<B: Backend>(
    values: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    not_dones: Tensor<B, 2>,
    gamma: f32,
    lambda: f32,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let mut returns = vec![0.0_f32; rewards.shape().num_elements()];
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

// TODO: rest of training
