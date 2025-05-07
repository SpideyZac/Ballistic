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

fn calculate_index(shape: &[usize], indicies: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;
    for (i, dim) in shape.iter().enumerate() {
        index += indicies[i] * stride;
        stride *= *dim;
    }
    index
}

pub struct FloatTensorView<'a, B: Backend, const D: usize> {
    pub tensor: &'a Tensor<B, D>,
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl<'a, B: Backend, const D: usize> FloatTensorView<'a, B, D> {
    pub fn new(tensor: &'a Tensor<B, D>) -> Self {
        let shape = tensor.shape();
        let data = tensor.to_data();
        let data = data.to_vec().unwrap();
        Self {
            tensor,
            shape,
            data,
        }
    }

    pub fn get(&self, indicies: &[usize]) -> f32 {
        let index = calculate_index(&self.shape.dims, indicies);

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

        for i in start_index..end_index {
            slice.push(self.data[i]);
        }

        slice
    }

    pub fn set_slice(&mut self, start_indicies: &[usize], end_indicies: &[usize], values: &[f32]) {
        let start_index = calculate_index(&self.shape.dims, start_indicies);
        let end_index = calculate_index(&self.shape.dims, end_indicies);

        for (i, value) in (start_index..end_index).zip(values.iter()) {
            self.data[i] = *value;
        }
    }

    // TODO: create a to_tensor method to convert the data back to a tensor
}
