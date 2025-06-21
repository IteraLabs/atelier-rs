/// Convex Linear Models
use std::f64;
use tch::{Device, Kind, TchError, Tensor};

pub trait Model {
    fn id(&mut self, id: String);
    fn forward(&self, features: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn save_model(&self, file_path: &str) -> Result<(), TchError>;
    fn load_model(&mut self, file_path: &str) -> Result<(), TchError>;
}

#[derive(Debug)]
pub struct LinearModel {
    pub id: String,
    pub weights: Tensor,
    pub bias: Tensor,
}

impl LinearModel {
    pub fn new(input_dim: i64) -> LinearModelBuilder {
        LinearModelBuilder::new(input_dim)
    }

    pub fn shallow_clone(&self) -> Self {
        LinearModel {
            id: self.id.clone(),
            weights: self.weights.shallow_clone(),
            bias: self.bias.shallow_clone(),
        }
    }

    pub fn set_weights(&mut self, weights: Tensor) {
        self.weights = weights;
    }

    pub fn set_bias(&mut self, bias: Tensor) {
        self.bias = bias;
    }

    pub fn forward_with_params(
        &self,
        x: &Tensor,
        weights: &Tensor,
        bias: &Tensor
    ) -> Tensor {

        x.matmul(weights) + bias

    }

}

impl Model for LinearModel {
    fn id(&mut self, id: String) {
        self.id = id;
    }

    fn forward(&self, features: &Tensor) -> Tensor {
        let fw = features.matmul(&self.weights).to_kind(Kind::Float)
            + &self.bias.to_kind(Kind::Float);

        fw.to_kind(Kind::Float)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    fn save_model(&self, file_path: &str) -> Result<(), TchError> {
        // Create a state dictionary compatible with PyTorch
        let state_dict = vec![
            ("weights".to_string(), self.weights.shallow_clone()),
            ("bias".to_string(), self.bias.shallow_clone()),
        ];

        // Save using PyTorch-compatible format
        tch::Tensor::save_multi(&state_dict, file_path)
    }

    fn load_model(&mut self, file_path: &str) -> Result<(), TchError> {
        // Load state dictionary
        let state_dict = tch::Tensor::load_multi(file_path)?;

        // Update model parameters
        for (name, tensor) in state_dict {
            match name.as_str() {
                "weight" => self.weights = tensor,
                "bias" => self.bias = tensor,
                _ => {
                    return Err(tch::TchError::FileFormat(format!(
                        "Unexpected tensor: {}",
                        name
                    )));
                }
            }
        }
        Ok(())
    }
}

impl Default for LinearModel {

    fn default() -> Self {
        LinearModel {
            id: String::new(),
            weights: Tensor::new(),
            bias: Tensor::new(),
        }
    }

}

#[derive(Debug)]
pub struct LinearModelBuilder {
    id: Option<String>,
    input_dim: i64,
    device: Device,
}

impl LinearModelBuilder {
    pub fn new(input_dim: i64) -> Self {
        LinearModelBuilder {
            id: None,
            input_dim,
            device: Device::Cpu,
        }
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn glorot_uniform_init(self) -> LinearModel {
        // Initialize weights with Glorot uniform
        let limit = (6.0_f64).sqrt() / ((self.input_dim + 1) as f64).sqrt();

        let rand_weights =
            Tensor::rand(&[self.input_dim], (Kind::Float, self.device)) * limit;

        // Initialize weights to "normalized initialization" factor.
        let weights = rand_weights.set_requires_grad(true);

        // Initialize bias to zeros
        let bias =
            Tensor::zeros(&[1], (Kind::Float, self.device)).set_requires_grad(true);

        LinearModel {
            id: self.id.unwrap_or_default(),
            weights,
            bias,
        }
    }
}
