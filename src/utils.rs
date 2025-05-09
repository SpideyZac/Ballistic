use std::marker::PhantomData;

use burn::{prelude::*, tensor::Shape};

pub fn tensor_std<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, 1> {
    let flat = tensor.flatten(0, D - 1);
    let mean = flat.clone().mean();

    let diff = flat.clone() - mean;
    let sq = diff.clone() * diff;

    let sum_sq = sq.sum();

    let n = (flat.shape().num_elements() as f32).max(1.0);
    let var_unbiased = sum_sq / (n - 1.0);

    var_unbiased.sqrt()
}

pub fn calculate_index(shape: &[usize], indicies: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;
    for (i, dim) in shape.iter().enumerate().rev() {
        index += indicies[i] * stride;
        stride *= *dim;
    }
    index
}

pub fn flat_to_tensor<B: Backend, const D: usize>(
    data: &[f32],
    shape: &[usize],
    device: &B::Device,
) -> Tensor<B, D> {
    let flat_tensor: Tensor<B, 1> = Tensor::from_floats(data, device);
    flat_tensor.reshape::<D, [usize; D]>(
        shape
            .try_into()
            .expect("Shape dimensions do not match tensor dimensions"),
    )
}

pub struct FloatTensorView<B: Backend, const D: usize> {
    pub shape: Shape,
    pub data: Vec<f32>,
    pub backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> FloatTensorView<B, D> {
    pub fn new(tensor: &Tensor<B, D>) -> Self {
        let shape = tensor.shape();
        let data = tensor.to_data();
        let data = data.to_vec().unwrap();
        Self {
            shape,
            data,
            backend: PhantomData,
        }
    }

    pub fn get(&self, indicies: &[usize]) -> f32 {
        let index = calculate_index(&self.shape.dims, indicies);
        println!("Index: {}", index);

        self.data[index]
    }

    pub fn set(&mut self, indicies: &[usize], value: f32) {
        let index = calculate_index(&self.shape.dims, indicies);

        self.data[index] = value;
    }

    pub fn get_slice(&self, start_indicies: &[usize], end_indicies: &[usize]) -> Vec<f32> {
        let mut slice = Vec::with_capacity(
            end_indicies
                .iter()
                .zip(start_indicies)
                .map(|(end, start)| end - start)
                .product(),
        );

        let start_index = calculate_index(&self.shape.dims, start_indicies);
        let end_index = calculate_index(&self.shape.dims, end_indicies);

        for i in start_index..=end_index {
            slice.push(self.data[i]);
        }

        slice
    }

    pub fn set_slice(&mut self, start_indicies: &[usize], end_indicies: &[usize], values: &[f32]) {
        let start_index = calculate_index(&self.shape.dims, start_indicies);
        let end_index = calculate_index(&self.shape.dims, end_indicies);

        for (i, value) in (start_index..=end_index).zip(values.iter()) {
            self.data[i] = *value;
        }
    }

    pub fn to_tensor(&self, device: &B::Device) -> Tensor<B, D> {
        flat_to_tensor(&self.data, &self.shape.dims, device)
    }
}
