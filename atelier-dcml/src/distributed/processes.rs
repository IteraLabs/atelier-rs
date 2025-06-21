use crate::{
    functions,
    metrics, models,
    optimizers,
};

use atelier_data::data;
use serde::Deserialize;
use std::{error::Error, fs};

use tch::{Tensor, Kind, Device};

#[derive(Debug)]
pub enum UpdateStrategy {
    CombineThenAdapt,
    AdaptThenCombine,
}

#[derive(Debug, Deserialize)]
struct Connection {
    from: usize,
    to: usize,
    weight: f64,
}

#[derive(Debug, Deserialize)]
struct Training {
    agents: u32,
    agent_connections: Vec<Connection>,
}

#[derive(Debug, Deserialize)]
struct TrainingTemplate {
    training: Vec<Training>,
}

#[derive(Debug)]
pub struct ConnectionsMatrix {
    values: Vec<f64>,
    n_agents: usize,
}

impl ConnectionsMatrix {

    pub fn new(n_agents: usize) -> Self {

        let matrix_values = vec![0.0; n_agents * n_agents];

        ConnectionsMatrix {
            values: matrix_values,
            n_agents,
        }

    }

    pub fn fill(mut self, connections_file: &str) -> Result<Self, Box<dyn Error>> {

        // Read and parse the TOML file
        let config_str = fs::read_to_string(connections_file)?;
        let template: TrainingTemplate = toml::from_str(&config_str)?;

        // Fill in the matrix with connection weights
        if let Some(training) = template.training.first() {
            for conn in &training.agent_connections {
                let from = conn.from;
                let to = conn.to;
                
                if from < self.n_agents && to < self.n_agents {
                    let index = from * self.n_agents + to;
                    self.values[index] = conn.weight;
                }
            }
        }
        
        // Ensure row-stochastic property for consensus
        self.normalize_rows();
        Ok(self)
    }

    fn normalize_rows(&mut self) {
        for i in 0..self.n_agents {
            let mut row_sum = 0.0;
            for j in 0..self.n_agents {
                row_sum += self.values[i * self.n_agents + j]; 
            }
            if row_sum > 0.0 {
                for j in 0..self.n_agents {
                    self.values[i * self.n_agents + j] /= row_sum;
                }
            } else {
                self.values[i * self.n_agents + i] = 1.0;
            }
        }
    }

    pub fn get_weight(&self, from: usize, to: usize) -> f64 {
        if from < self.n_agents && to < self.n_agents {
            self.values[from * self.n_agents + to]
        } else {
            0.0
        }
    }

    pub fn to_tensor(&self, device: Device) -> Tensor {
        Tensor::from_slice(&self.values)
        .view([self.n_agents as i64, self.n_agents as i64])
        .to(device)
    }
    
}

#[derive(Debug)]
pub struct Distributed {
    v_datasets: Vec<data::Dataset>,
    v_models: Vec<models::LinearModel>,
    v_losses: Vec<functions::CrossEntropy>,
    v_metrics: Vec<metrics::Metrics>,
    optimizer: optimizers::GradientDescent,
    topology: ConnectionsMatrix,
    strategy: UpdateStrategy,
    learning_rate: f64,
}

impl Distributed {

    pub fn new() -> DistributedBuilder {

        DistributedBuilder::new()
    
    }

    // --------------------------------------------------------------------- TRAIN --- //

    pub fn train(&mut self, epochs: usize) -> Result<(), Box<dyn Error>> {

        let n_agents = self.v_models.len();
        let device = self.v_models[0].weights.device();

        for epoch in 0..epochs {

            match self.strategy {
                UpdateStrategy::CombineThenAdapt => {
                    self.combine_then_adapt_step(n_agents, device)?;
                }
                UpdateStrategy::AdaptThenCombine => {
                    self.adapt_then_combine_step(n_agents, device)?;
                }
            }

            if epoch % 1 == 0 {
                self.log_metrics(epoch)?;
            }

        }

        Ok(())

    }

    // ----------------------------------------------------------- STRATEGY 1: CTA --- //
    
    
    fn combine_then_adapt_step(
        &mut self,
        n_agents: usize,
        _device: Device
    ) -> Result<(), Box<dyn Error>> {
        // Step 1: Combine (Consensus) - collect all model states first
        let model_weights: Vec<Tensor> = self.v_models.iter()
            .map(|m| m.weights.shallow_clone())
            .collect();
        let model_biases: Vec<Tensor> = self.v_models.iter()
            .map(|m| m.bias.shallow_clone())
            .collect();
        
        let mut mixed_weights = Vec::new();
        let mut mixed_biases = Vec::new();

        for i in 0..n_agents {
            let mut combined_weight = Tensor::zeros_like(&model_weights[i]);
            let mut combined_bias = Tensor::zeros_like(&model_biases[i]);

            // Weighted combination based on topology
            for j in 0..n_agents {
                let weight_ij = self.topology.get_weight(i, j);
                if weight_ij > 0.0 {
                    combined_weight += &model_weights[j] * weight_ij;
                    combined_bias += &model_biases[j] * weight_ij;
                }
            }

            mixed_weights.push(combined_weight);
            mixed_biases.push(combined_bias);
        }

        // Step 2: Adapt - Gradient descent on mixed variables
        for i in 0..n_agents {
            let (features, targets) = self.v_datasets[i].clone().from_vec_to_tensor();

            // Create leaf tensors for gradient computation
            let mut weight_var = mixed_weights[i].detach().requires_grad_(true);
            let mut bias_var = mixed_biases[i].detach().requires_grad_(true);

            // Forward pass with mixed weights
            let linear_output = features.matmul(&weight_var) + &bias_var;
            let y_hat = linear_output.sigmoid(); // or appropriate activation

            // Compute loss
            let loss = self.v_losses[i].compute_loss(&y_hat, &targets);

            // Backward pass
            loss.backward();

            // Get gradients
            let weight_grad = weight_var.grad();
            let bias_grad = bias_var.grad();

            // Update actual model weights using no_grad
            tch::no_grad(|| {
                self.v_models[i].weights = &mixed_weights[i] - &weight_grad * self.learning_rate;
                self.v_models[i].bias = &mixed_biases[i] - &bias_grad * self.learning_rate;
            });

            // Clear gradients
            weight_var.zero_grad();
            bias_var.zero_grad();
        }

        Ok(())
    }
    
    // ----------------------------------------------------------- STRATEGY 1: ATC --- //

    fn adapt_then_combine_step(
        &mut self,
        n_agents: usize,
        _device: Device
    ) -> Result<(), Box<dyn Error>> {
        // Step 1: Adapt - Local gradient descent first
        let mut adapted_weights = Vec::new();
        let mut adapted_biases = Vec::new();

        for i in 0..n_agents {
            let (features, targets) = self.v_datasets[i].clone().from_vec_to_tensor();

            // Enable gradients and retain for non-leaf tensors
            let _ = self.v_models[i].weights.requires_grad_(true);
            let _ = self.v_models[i].bias.requires_grad_(true);

            // Forward pass with retained gradients
            let linear_output = features.matmul(&self.v_models[i].weights) + &self.v_models[i].bias;
            let y_hat = linear_output.sigmoid();

            // Compute loss
            let loss = self.v_losses[i].compute_loss(&y_hat, &targets);

            // Backward pass with retained gradients
            y_hat.retains_grad();
            loss.backward();

            // Get gradients with defined() check
            let weight_grad = self.v_models[i].weights.grad();
            let bias_grad = self.v_models[i].bias.grad();

            // Validate gradients before use
            if !weight_grad.defined() || !bias_grad.defined() {
                return Err("Undefined gradients after backward pass".into());
            }

            // Compute adapted weights in no_grad block
            tch::no_grad(|| {
                let adapted_weight = &self.v_models[i].weights - &weight_grad * self.learning_rate;
                let adapted_bias = &self.v_models[i].bias - &bias_grad * self.learning_rate;

                adapted_weights.push(adapted_weight);
                adapted_biases.push(adapted_bias);

                // Reset gradients safely
                self.v_models[i].weights.zero_grad();
                self.v_models[i].bias.zero_grad();
            });
        }

        // Step 2: Combine - Consensus on adapted weights
        for i in 0..n_agents {
            let mut combined_weight = Tensor::zeros_like(&adapted_weights[i]);
            let mut combined_bias = Tensor::zeros_like(&adapted_biases[i]);

            // Weighted combination in no_grad block
            tch::no_grad(|| {
                for j in 0..n_agents {
                    let weight_ij = self.topology.get_weight(i, j);
                    if weight_ij > 0.0 {
                        combined_weight += &adapted_weights[j] * weight_ij;
                        combined_bias += &adapted_biases[j] * weight_ij;
                    }
                }
                
                // Update model weights
                self.v_models[i].weights = combined_weight;
                self.v_models[i].bias = combined_bias;
            });
        }

        Ok(())
    }
    fn log_metrics(&mut self, epoch: usize) -> Result<(), Box<dyn Error>> {
        let mut v_losses = Vec::new();
        let mut v_accuracies = Vec::new();

        for i in 0..self.v_models.len() {
            let (features, targets) = self.v_datasets[i].clone().from_vec_to_tensor();
            
            // Forward pass for evaluation
            let linear_output = features.matmul(&self.v_models[i].weights) + &self.v_models[i].bias;
            let y_hat = linear_output.sigmoid();
            
            // Compute loss
            let loss = self.v_losses[i].compute_loss(&y_hat, &targets);
            
            // Manual accuracy calculation to ensure correctness
            let predictions = y_hat.ge(0.5).to_kind(Kind::Float);
            let correct = predictions.eq_tensor(&targets).sum(Kind::Float);
            let total = targets.size()[0] as f64;
            let accuracy = correct.double_value(&[]) / total;
            
            // Ensure accuracy is in valid range [0, 1]
            let clamped_accuracy = accuracy.max(0.0).min(1.0);
            
            v_losses.push(loss.double_value(&[]));
            v_accuracies.push(clamped_accuracy);
        }

        println!(
            "\nEpoch {}\nStrategy: {:?}\nLosses: {:?}\nAccuracies: {:?}\n",
            epoch, self.strategy, v_losses, v_accuracies
        );

        Ok(())
    }
}

pub struct DistributedBuilder {
    v_datasets: Option<Vec<data::Dataset>>,
    v_models: Option<Vec<models::LinearModel>>,
    v_losses: Option<Vec<functions::CrossEntropy>>,
    v_metrics: Option<Vec<metrics::Metrics>>,
    optimizer: Option<optimizers::GradientDescent>,
    topology: Option<ConnectionsMatrix>,
    strategy: Option<UpdateStrategy>,
    learning_rate: Option<f64>,
}

impl DistributedBuilder {
    pub fn new() -> Self {
        DistributedBuilder {
            v_datasets: None,
            v_models: None,
            v_losses: None,
            v_metrics: None,
            optimizer: None,
            topology: None,
            strategy: None,
            learning_rate: None,
        }
    }

    pub fn datasets(mut self, datasets: Vec<data::Dataset>) -> Self {
        self.v_datasets = Some(datasets);
        self
    }

    pub fn models(mut self, models: Vec<models::LinearModel>) -> Self {
        self.v_models = Some(models);
        self
    }

    pub fn losses(mut self, losses: Vec<functions::CrossEntropy>) -> Self {
        self.v_losses = Some(losses);
        self
    }

    pub fn metrics(mut self, metrics: Vec<metrics::Metrics>) -> Self {
        self.v_metrics = Some(metrics);
        self
    }

    pub fn optimizer(mut self, optimizer: optimizers::GradientDescent) -> Self {
        self.optimizer = Some(optimizer);
        self
    }
    
    pub fn topology(mut self, topology: ConnectionsMatrix) -> Self {
        self.topology = Some(topology);
        self
    }

    pub fn strategy(mut self, strategy: UpdateStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    pub fn build(self) -> Result<Distributed, &'static str> {
        let v_datasets = self.v_datasets.ok_or("Missing datasets")?;
        let v_models = self.v_models.ok_or("Missing models")?;
        let v_losses = self.v_losses.ok_or("Missing losses")?;
        let v_metrics = self.v_metrics.ok_or("Missing metrics")?;
        let optimizer = self.optimizer.ok_or("Missing optimizer")?;
        let topology = self.topology.ok_or("Missing topology")?;
        let strategy = self.strategy.ok_or("Missing strategy")?;
        let learning_rate = self.learning_rate.unwrap_or(0.01);

        // Validate dimensions
        if v_datasets.len() != v_models.len() || 
           v_models.len() != v_losses.len() || 
           v_losses.len() != v_metrics.len() {
            return Err("Mismatched vector lengths");
        }

        Ok(Distributed {
            v_datasets,
            v_models,
            v_losses,
            v_metrics,
            optimizer,
            topology,
            strategy,
            learning_rate,
        })
    }
}
