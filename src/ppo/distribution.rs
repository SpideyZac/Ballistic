use std::f32::consts::PI;

use burn::{prelude::*, tensor::Distribution};

pub struct NormalDistribution<'a, B: Backend> {
    mean: &'a Tensor<B, 2>,
    std: &'a Tensor<B, 2>,
    device: &'a B::Device,
}

impl<'a, B: Backend> NormalDistribution<'a, B> {
    pub fn new(mean: &'a Tensor<B, 2>, std: &'a Tensor<B, 2>, device: &'a B::Device) -> Self {
        Self { mean, std, device }
    }

    pub fn sample(&self) -> Tensor<B, 2> {
        let eps = Tensor::random(
            self.mean.shape(),
            Distribution::Normal(0.0, 1.0),
            self.device,
        );

        self.mean.clone() + self.std.clone() * eps
    }

    pub fn log_prob(&self, action: &Tensor<B, 2>) -> Tensor<B, 2> {
        let var = self.std.clone().powf_scalar(2.0);
        let log_scale = self.std.clone().log();

        -((action.clone() - self.mean.clone()).powf_scalar(2.0)) / var.mul_scalar(2.0)
            - log_scale
            - (2.0 * PI).sqrt().ln()
    }

    pub fn entropy(&self) -> Tensor<B, 2> {
        self.std.clone().log() + (0.5 + 0.5 * (2.0 * PI).ln())
    }
}
