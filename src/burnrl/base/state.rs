use std::fmt::Debug;

use burn::tensor::{Tensor, backend::Backend};

pub trait State: Debug + Copy + Clone {
    type Data;
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1>;

    fn size() -> usize;
}
