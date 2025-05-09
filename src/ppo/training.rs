use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::{rng, seq::SliceRandom};

use crate::{
    ppo::{config::PPOTrainingConfig, model::Agent, rollout_buffer::RolloutBufferBatch},
    utils::{FloatTensorView, flat_to_tensor, tensor_std},
};

pub fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub struct TrainingBatch<B: Backend> {
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
            / (tensor_std(advantages_flat) + 1e-8);
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

pub fn train_step<B: AutodiffBackend>(
    mut model: Agent<B>,
    optimizer: &mut (impl Optimizer<Agent<B>, B> + Sized),
    train_batch: &TrainingBatch<B>,
    device: &B::Device,
    config: &PPOTrainingConfig,
) -> Agent<B> {
    let obs_view = FloatTensorView::new(&train_batch.obs);
    let actions_view = FloatTensorView::new(&train_batch.actions);
    let log_probs_view = FloatTensorView::new(&train_batch.log_probs);
    let advantages_view = FloatTensorView::new(&train_batch.advantages);
    let returns_view = FloatTensorView::new(&train_batch.returns);
    let values_view = FloatTensorView::new(&train_batch.values);

    let mut rng = rng();
    let batch_indicies = Tensor::<B, 1, Int>::arange(0..config.batch_size as i64, device);
    let batch_indicies_data = batch_indicies.to_data();
    let batch_indicies: Vec<i64> = batch_indicies_data.to_vec().unwrap();
    let mut batch_indicies: Vec<usize> = batch_indicies.iter().map(|&x| x as usize).collect();

    for epoch in 0..config.update_epochs {
        // TODO: better debugging
        println!("Epoch: {}", epoch);
        batch_indicies.shuffle(&mut rng);
        let mut approx_kl = Tensor::<B, 1>::zeros([1], device);
        for start in (0..config.batch_size).step_by(config.mini_batch_size) {
            let end = start + config.mini_batch_size;

            let batch_observations = flat_to_tensor(
                obs_view
                    .get_slice(&[start, 0], &[end, config.obs_dim])
                    .as_slice(),
                &[end - start + 1, config.obs_dim],
                device,
            );
            let batch_actions = flat_to_tensor(
                actions_view
                    .get_slice(&[start, 0], &[end, config.action_dim])
                    .as_slice(),
                &[end - start + 1, config.action_dim],
                device,
            );
            let (_, new_log_probs, entropies, new_values) =
                model.get_action_and_value(batch_observations, Some(batch_actions));
            let batch_log_probs: Tensor<B, 1> = flat_to_tensor(
                log_probs_view.get_slice(&[start], &[end]).as_slice(),
                &[end - start + 1],
                device,
            );
            // TODO: only calculate approx_kl if its the last minibatch
            let log_ratio = new_log_probs - batch_log_probs;
            let ratio = log_ratio.clone().exp();

            approx_kl = ((ratio.clone() - 1.0) - log_ratio).mean();

            let batch_advantages: Tensor<B, 1> = flat_to_tensor(
                advantages_view.get_slice(&[start], &[end]).as_slice(),
                &[end - start + 1],
                device,
            );
            let pg_loss1 = -batch_advantages.clone() * ratio.clone();
            let pg_loss2 = -batch_advantages.clone()
                * ratio
                    .clone()
                    .clamp(1.0 - config.clip_coef, 1.0 + config.clip_coef);
            let pg_loss = Tensor::<B, 2>::max_dim(Tensor::stack(vec![pg_loss1, pg_loss2], 0), 0)
                .squeeze::<1>(0)
                .mean();

            let batch_returns: Tensor<B, 1> = flat_to_tensor(
                returns_view.get_slice(&[start], &[end]).as_slice(),
                &[end - start + 1],
                device,
            );
            let batch_values: Tensor<B, 1> = flat_to_tensor(
                values_view.get_slice(&[start], &[end]).as_slice(),
                &[end - start + 1],
                device,
            );
            let new_values: Tensor<B, 1> = new_values.flatten(0, 1);
            let v_loss = if config.clip_vloss {
                let v_loss_unclipped =
                    (new_values.clone() - batch_returns.clone()).powf_scalar(2.0);
                let v_clipped = batch_values.clone()
                    + (new_values - batch_values.clone())
                        .clamp(-config.clip_coef, config.clip_coef);
                let v_loss_clipped = (v_clipped - batch_returns).powf_scalar(2.0);
                let v_loss_max = Tensor::<B, 2>::max_dim(
                    Tensor::stack(vec![v_loss_unclipped, v_loss_clipped], 0),
                    0,
                )
                .squeeze::<1>(0);
                v_loss_max.mean() * 0.5
            } else {
                ((new_values - batch_returns).powf_scalar(2.0)).mean() * 0.5
            };

            let entropy_loss = entropies.mean();
            let loss = pg_loss - entropy_loss * config.ent_coef + v_loss * config.vf_coef;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.lr as f64, model, grads_params);
        }

        if let Some(target_kl) = config.target_kl {
            let data = approx_kl.to_data();
            let data = data.to_vec::<f32>().unwrap();
            let approx_kl = data[0];
            if approx_kl > target_kl {
                break;
            }
        }
    }

    model
}

// TODO: run env
