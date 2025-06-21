use crate::{
    functions,
    functions::{RegType, Regularized},
    metrics, models,
    models::Model,
    optimizers,
    optimizers::Optimizer,
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

        // --- Step 1: Combine (Consensus)
        let mut mixed_weights = Vec::new();
        let mut mixed_biases = Vec::new();

        for i in 0..n_agents {

            let mut combined_weight = Tensor::zeros_like(&self.v_models[i].weights);
            let mut combined_bias = Tensor::zeros_like(&self.v_models[i].bias);

            // Weighted combination based on topology
            for j in 0..n_agents {

                let weight_ij = self.topology.get_weight(i, j);
                if weight_ij > 0.0 {
                    combined_weight += &self.v_models[j].weights * weight_ij;
                    combined_bias += &self.v_models[j].bias * weight_ij;
                }

            }

            mixed_weights.push(combined_weight);
            mixed_biases.push(combined_bias);

        }

        // --- Step 2: Adapt - Gradient descent on mixed variables
        for i in 0..n_agents {

            // Dataset extraction for model
            let (features, targets) = self
                .v_datasets[i]
                .clone()
                .from_vec_to_tensor();

            // Create new leaf tensors with gradient tracking
            let mut weight_var = mixed_weights[i].detach().requires_grad_(true);
            let mut bias_var = mixed_biases[i].detach().requires_grad_(true);

            // Forward pass with mixed weights
            let y_hat = self
                .v_models[i]
                .forward_with_params(&features, &weight_var, &bias_var);

            // Compute loss
            let loss = self
                .v_losses[i]
                .compute_loss(&y_hat, &targets);

            let reg_param_c = 5.0;
            let reg_param_lambda = 0.9;

            let reg_loss = self
                .v_losses[i]
                .regularize(
                    &weight_var,
                    &RegType::Elasticnet,
                    vec![reg_param_c, reg_param_lambda],
                )
                .sum(Kind::Float);

            let total_loss = &loss + &reg_loss;
            total_loss.backward();

            // Update weights using gradients
            let c_w = weight_var.grad();
            let c_b = bias_var.grad();

            // Temporarily remove the model to mutate it
            let mut model = std::mem::take(&mut self.v_models[i]);
            
            // Apply optimization step
            self.optimizer.step(
                &mut model.weights,
                &mut model.bias,
                &c_w, 
                &c_b,
            );
            
            // Put model back
            self.v_models[i] = model;
            
            // Zero gradients for next iteration
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
    
        let mut adapted_weights = Vec::new();
        let mut adapted_biases = Vec::new();

        // --- Step 1: Adapt - Gradient descent on local variables
        for i in 0..n_agents {

            // Dataset extraction for model
            let (features, targets) = &self
                .v_datasets[i]
                .clone()
                .from_vec_to_tensor();

            // Enable gradients
            let _ = self.v_models[i].weights.requires_grad_(true);
            let _ = self.v_models[i].bias.requires_grad_(true);

            // Forward pass
            let y_hat = self
                .v_models[i]
                .forward(&features);

            // Compute loss
            let loss = self
                .v_losses[i]
                .compute_loss(&y_hat, &targets);

            let reg_param_c = 1.9;
            let reg_param_lambda = 0.8;
            
            // Add regularization
            let reg_loss = self
                .v_losses[i]
                .regularize(
                    &self.v_models[i].weights,
                    &RegType::Elasticnet,
                    vec![reg_param_c, reg_param_lambda],
                )
                .sum(Kind::Float);

            let total_loss = &loss + &reg_loss;
            total_loss.backward();

            // Compute gradients
            let c_w = self.v_models[i].weights.grad();
            let c_b = self.v_models[i].bias.grad();

            // Local adaptation step
            let adapted_weight = &self.v_models[i].weights - &c_w * self.learning_rate;
            let adapted_bias = &self.v_models[i].bias - &c_b * self.learning_rate;

            adapted_weights.push(adapted_weight);
            adapted_biases.push(adapted_bias);

            // Zero gradients
            self.v_models[i].weights.zero_grad();
            self.v_models[i].bias.zero_grad();
        }

        // Step 2: Combine - Consensus on adapted variables
        for i in 0..n_agents {

            let mut combined_weight = Tensor::zeros_like(&adapted_weights[i]);
            let mut combined_bias = Tensor::zeros_like(&adapted_biases[i]);

            // Weighted combination based on topology
            for j in 0..n_agents {

                let weight_ij = self.topology.get_weight(i, j);
                if weight_ij > 0.0 {
                    combined_weight += &adapted_weights[j] * weight_ij;
                    combined_bias += &adapted_biases[j] * weight_ij;
                }

            }

            // Update model with combined weights
            self.v_models[i].weights = combined_weight;
            self.v_models[i].bias = combined_bias;
        }

        Ok(())
    }

    fn log_metrics(&mut self, epoch: usize) -> Result<(), Box<dyn Error>> {

        let mut v_losses: Vec<f64> = Vec::new();
        let mut v_accuracies: Vec<f64> = Vec::new();

        for i in 0..self.v_models.len() {

            let (features, targets) = &self.v_datasets[i].clone().from_vec_to_tensor();
            let y_hat = self.v_models[i].forward(&features);
            let loss = self.v_losses[i].compute_loss(&y_hat, &targets);
            let metrics = self.v_metrics[i].compute_all(&y_hat, &targets);

            println!("metrics[{:?}] {:?}", metrics, i);

            let loss_value: f64 = loss.double_value(&[]);
            v_accuracies.push(metrics["accuracy"].as_scalar().unwrap());
            v_losses.push(loss_value);

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
