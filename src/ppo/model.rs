use core::f64::consts::SQRT_2;

use burn::{
    module::{Param, ParamId},
    prelude::*,
};

use crate::ppo::distribution::NormalDistribution;

fn layer_init<B: Backend>(layer: nn::LinearConfig, std_: f64, device: &B::Device) -> nn::Linear<B> {
    layer
        .with_bias(true)
        .with_initializer(nn::Initializer::XavierUniform { gain: std_ })
        .init(device)
}

#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    pub linear1: nn::Linear<B>,
    pub linear2: nn::Linear<B>,
    pub linear3: nn::Linear<B>,
}

#[derive(Config)]
pub struct CriticConfig {
    pub obs_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
}

impl CriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Critic<B> {
        let linear1 = layer_init(
            nn::LinearConfig::new(self.obs_dim, self.hidden_dim),
            SQRT_2,
            device,
        );
        let linear2 = layer_init(
            nn::LinearConfig::new(self.hidden_dim, self.hidden_dim),
            SQRT_2,
            device,
        );
        let linear3 = layer_init(nn::LinearConfig::new(self.hidden_dim, 1), 1.0, device);

        Critic {
            linear1,
            linear2,
            linear3,
        }
    }
}

impl<B: Backend> Critic<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x).tanh();
        let x = self.linear2.forward(x).tanh();

        self.linear3.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    pub linear1: nn::Linear<B>,
    pub linear2: nn::Linear<B>,
    pub linear3: nn::Linear<B>,
    pub log_std: Param<Tensor<B, 2>>,
}

#[derive(Config)]
pub struct ActorConfig {
    pub obs_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
    pub action_dim: usize,
}

impl ActorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Actor<B> {
        let linear1 = layer_init(
            nn::LinearConfig::new(self.obs_dim, self.hidden_dim),
            SQRT_2,
            device,
        );
        let linear2 = layer_init(
            nn::LinearConfig::new(self.hidden_dim, self.hidden_dim),
            SQRT_2,
            device,
        );
        let linear3 = layer_init(
            nn::LinearConfig::new(self.hidden_dim, self.action_dim),
            0.01,
            device,
        );

        let action_dim = self.action_dim;
        let log_std = Param::uninitialized(
            ParamId::new(),
            move |device, require_grad| {
                let mut tensor = Tensor::zeros([1, action_dim], device);
                if require_grad {
                    tensor = tensor.require_grad();
                }
                tensor
            },
            device.clone(),
            true,
        );

        Actor {
            linear1,
            linear2,
            linear3,
            log_std,
        }
    }
}

impl<B: Backend> Actor<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x).tanh();
        let x = self.linear2.forward(x).tanh();

        self.linear3.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Agent<B: Backend> {
    actor: Actor<B>,
    critic: Critic<B>,
}

#[derive(Config)]
pub struct AgentConfig {
    pub obs_dim: usize,
    #[config(default = 64)]
    pub actor_hidden_dim: usize,
    #[config(default = 64)]
    pub critic_hidden_dim: usize,
    pub action_dim: usize,
}

impl AgentConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Agent<B> {
        let actor_config = ActorConfig {
            obs_dim: self.obs_dim,
            hidden_dim: self.actor_hidden_dim,
            action_dim: self.action_dim,
        };
        let critic_config = CriticConfig {
            obs_dim: self.obs_dim,
            hidden_dim: self.critic_hidden_dim,
        };

        let actor = actor_config.init(device);
        let critic = critic_config.init(device);

        Agent { actor, critic }
    }
}

impl<B: Backend> Agent<B> {
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

        if action.is_none() {
            action = Some(sampled_action);
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
