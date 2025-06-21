
use crate::{
    functions,
    functions::{RegType, Regularized},
    metrics, models,
    models::Model,
    optimizers,
    optimizers::Optimizer,
};

use atelier_data::data;
use std::error::Error;

use tch::Kind;

pub enum TrainType {
    Batch,
}

///
/// Singular training
///
/// data: Features, Target
/// model: Model definition
/// loss: A function used as loss function, should implement compute_loss 
/// optimizer: Optimization/Learning Algorithm for weight updates
/// metrics: Learning Performance, and others.

#[derive(Debug)]
pub struct Singular {
    dataset: data::Dataset,
    model: models::LinearModel,
    loss: functions::CrossEntropy,
    optimizer: optimizers::GradientDescent,
    metrics: metrics::Metrics,
}

impl Singular {
    pub fn new() -> SingularBuilder {
        SingularBuilder::new()
    }

    pub fn train(&mut self, epochs: u32) -> Result<(), Box<dyn Error>> {
        let (features, targets) = &self.dataset.clone().from_vec_to_tensor();

        for epoch in 0..epochs {
            // --- Forward Step --- //
            let y_hat = self.model.forward(&features);

            // --- Compute Loss --- //
            let loss = self.loss.compute_loss(&y_hat, &targets);

            let reg_param_c = 1.9;
            let reg_param_lambda = 0.8;

            let reg_loss = self
                .loss
                .regularize(
                    &self.model.weights,
                    &RegType::Elasticnet,
                    vec![reg_param_c, reg_param_lambda],
                )
                .sum(Kind::Float);

            let total_loss = &loss + &reg_loss;
            total_loss.backward();

            // --- Compute Gradients --- //
            let c_w = self.model.weights.grad();
            let c_b = self.model.bias.grad();

            // --- Compute Step of Learning Algorithm
            self.optimizer.step(
                &mut self.model.weights,
                &mut self.model.bias,
                &c_w,
                &c_b,
            );

            // --- Reset gradient value on weights and bias
            self.model.weights.zero_grad();
            self.model.bias.zero_grad();

            // --- Get Metrics --- //
            let metrics = self.metrics.compute_all(&y_hat, &targets);

            println!(
                "\n--- epoch {:?} --- loss {:?} --- accuracy: {:?}",
                epoch, &loss, metrics["accuracy"]
            );
        }
        Ok(())
    }

    pub fn save_model(self, file_route: &str) {
        // --- Save a model's weight
        let _ = self.model.save_model(file_route);
    }

    pub fn load_model(mut self, file_route: &str) {
        // --- Load a model's weight
        let _ = self.model.load_model(file_route);
    }
}

pub struct SingularBuilder {
    dataset: Option<data::Dataset>,
    model: Option<models::LinearModel>,
    loss: Option<functions::CrossEntropy>,
    optimizer: Option<optimizers::GradientDescent>,
    metrics: Option<metrics::Metrics>,
}

impl SingularBuilder {
    pub fn new() -> Self {
        SingularBuilder {
            dataset: None,
            model: None,
            loss: None,
            optimizer: None,
            metrics: None,
        }
    }

    pub fn dataset(mut self, dataset: data::Dataset) -> Self {
        self.dataset = Some(dataset);
        self
    }

    pub fn model(mut self, model: models::LinearModel) -> Self {
        self.model = Some(model);
        self
    }

    pub fn loss(mut self, loss: functions::CrossEntropy) -> Self {
        self.loss = Some(loss);
        self
    }

    pub fn optimizer(mut self, optimizer: optimizers::GradientDescent) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    pub fn metrics(mut self, metrics: metrics::Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn build(self) -> Result<Singular, &'static str> {
        let dataset = self.dataset.ok_or("Missing data")?;
        let model = self.model.ok_or("MIssing model")?;
        let loss = self.loss.ok_or("Missing loss")?;
        let optimizer = self.optimizer.ok_or("Missing optimizer")?;
        let metrics = self.metrics.ok_or("Missing metrics")?;

        Ok(Singular {
            dataset,
            model,
            loss,
            optimizer,
            metrics,
        })
    }
}

