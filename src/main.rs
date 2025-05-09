mod ppo;
mod utils;

fn main() {
    rocketsim_rs::init(None, false);
    // let a = burn::tensor::Tensor::<burn::backend::NdArray<f32>, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
    // let mut view = utils::FloatTensorView::new(&a);
    // let elem = view.get(&[0, 1]); // 2.0
    // let slice = view.get_slice(&[0, 0], &[1, 1]); // [1.0, 2.0, 3.0, 4.0]
    // println!("Element: {}", elem);
    // println!("Slice: {:?}", slice);
    // view.set(&[0, 1], 5.0);
    // let tensor = view.to_tensor(&burn::backend::ndarray::NdArrayDevice::Cpu);
    // let data = tensor.to_data();
    // let data = data.to_vec::<f32>().unwrap();
    // println!("Updated Tensor: {:?}", data);
}
