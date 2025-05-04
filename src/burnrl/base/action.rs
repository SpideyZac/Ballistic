use std::fmt::Debug;

use rand::{Rng, rng};

pub trait Action: Debug + Copy + Clone + From<u32> + Into<u32> {
    fn random() -> Self {
        (rng().random_range(0..Self::size()) as u32).into()
    }

    fn enumerate() -> Vec<Self>;

    fn size() -> usize {
        Self::enumerate().len()
    }
}
