
use tch::{no_grad, Tensor};

pub trait Optimizer {
    fn id(&mut self, id: String);
    fn step(
        &self,
        weights: &mut Tensor,
        bias: &mut Tensor,
        weights_gradients: &Tensor,
        bias_gradients: &Tensor,
    );
    fn reset(&mut self);
}

#[derive(Debug)]
pub struct GradientDescent {
    id: String,
    learning_rate: f64,
}

impl GradientDescent {
    pub fn new() -> OptimizerBuilder {
        OptimizerBuilder::new()
    }
}

impl Optimizer for GradientDescent {
    fn id(&mut self, id: String) {
        self.id = id;
    }

    fn step(
        &self,
        weights: &mut Tensor,
        bias: &mut Tensor,
        weights_gradients: &Tensor,
        bias_gradients: &Tensor,
    ) {
        no_grad(|| {
            let _ = weights.f_sub_(&(weights_gradients * self.learning_rate));
            let _ = bias.f_sub_(&(bias_gradients * self.learning_rate));
        })
    }

    fn reset(&mut self) {}
}

pub struct OptimizerBuilder {
    id: Option<String>,
    learning_rate: Option<f64>,
}

impl OptimizerBuilder {
    pub fn new() -> Self {
        OptimizerBuilder {
            id: None,
            learning_rate: None,
        }
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn build(self) -> Result<GradientDescent, &'static str> {
        let id = self.id.ok_or("Missing id")?;
        let learning_rate = self.learning_rate.ok_or("Missing Learning Rate")?;
        Ok(GradientDescent { id, learning_rate })
    }
}
