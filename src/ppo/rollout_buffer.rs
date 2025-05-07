use burn::prelude::*;

pub struct RolloutBufferBatch<B: Backend> {
    pub obs: Tensor<B, 3>,
    pub actions: Tensor<B, 3>,
    pub rewards: Tensor<B, 2>,
    pub not_dones: Tensor<B, 2>,
    pub log_probs: Tensor<B, 2>,
    pub values: Tensor<B, 2>,
}

pub struct RolloutBuffer<B: Backend> {
    pub obs: Vec<Tensor<B, 2>>,
    pub actions: Vec<Tensor<B, 2>>,
    pub rewards: Vec<Tensor<B, 1>>,
    pub not_dones: Vec<Tensor<B, 1>>,
    pub log_probs: Vec<Tensor<B, 1>>,
    pub values: Vec<Tensor<B, 1>>,
}

impl<B: Backend> RolloutBuffer<B> {
    pub fn new(num_steps: usize) -> Self {
        Self {
            obs: Vec::with_capacity(num_steps),
            actions: Vec::with_capacity(num_steps),
            rewards: Vec::with_capacity(num_steps),
            not_dones: Vec::with_capacity(num_steps),
            log_probs: Vec::with_capacity(num_steps),
            values: Vec::with_capacity(num_steps),
        }
    }

    pub fn clear(&mut self) {
        self.obs.clear();
        self.actions.clear();
        self.rewards.clear();
        self.not_dones.clear();
        self.log_probs.clear();
        self.values.clear();
    }

    pub fn generate_batch(&self) -> RolloutBufferBatch<B> {
        let obs = Tensor::stack(self.obs.clone(), 0);
        let actions = Tensor::stack(self.actions.clone(), 0);
        let rewards = Tensor::stack(self.rewards.clone(), 0);
        let not_dones = Tensor::stack(self.not_dones.clone(), 0);
        let log_probs = Tensor::stack(self.log_probs.clone(), 0);
        let values = Tensor::stack(self.values.clone(), 0);

        RolloutBufferBatch {
            obs,
            actions,
            rewards,
            not_dones,
            log_probs,
            values,
        }
    }
}
