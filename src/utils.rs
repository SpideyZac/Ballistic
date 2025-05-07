use burn::prelude::*;

pub fn tensor_std<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Tensor<B, 1> {
    let flat = tensor.clone().flatten(0, D - 1);
    let mean = flat.clone().mean();

    let diff = flat.clone() - mean;
    let sq = diff.clone() * diff;

    let sum_sq = sq.sum();

    let n = (flat.shape().num_elements() as f32).max(1.0);
    let var_unbiased = sum_sq / (n - 1.0);

    var_unbiased.sqrt()
}
