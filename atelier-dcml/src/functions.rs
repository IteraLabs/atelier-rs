/// Loss Functions
use tch::{Kind, Tensor};

pub trait Regularized {
    fn id(&mut self, id: String);
    fn regularize(
        &self,
        weights: &Tensor,
        operation: &RegType,
        params: Vec<f64>,
    ) -> Tensor;
}

#[derive(Debug, Copy, Clone)]
pub enum RegType {
    L1,
    L2,
    Elasticnet,
}

#[derive(Debug)]
pub struct CrossEntropyBuilder<'a> {
    id: Option<&'a String>,
}

impl<'a> CrossEntropyBuilder<'a> {
    pub fn new() -> Self {
        CrossEntropyBuilder { id: None }
    }

    pub fn id(mut self, id: &'a String) -> Self {
        self.id = Some(&id);
        self
    }

    pub fn build(self) -> Result<CrossEntropy, &'static str> {
        let id = self.id.ok_or("Missing id value")?;

        Ok(CrossEntropy { id: id.to_string() })
    }
}

#[derive(Debug)]
pub struct CrossEntropy {
    pub id: String,
}

impl CrossEntropy {
    pub fn new<'a>() -> CrossEntropyBuilder<'a> {
        CrossEntropyBuilder::new()
    }

    pub fn compute_loss(&self, y_hat: &Tensor, y_true: &Tensor) -> Tensor {
        y_hat
            .binary_cross_entropy_with_logits::<&Tensor>(
                &y_true,
                None,                 // weight
                None,                 // pos_weight
                tch::Reduction::Mean, //
            )
            .set_requires_grad(true)
    }
}

impl Regularized for CrossEntropy {
    fn id(&mut self, id: String) {
        self.id = id;
    }

    fn regularize(
        &self,
        weights: &Tensor,
        operation: &RegType,
        params: Vec<f64>,
    ) -> Tensor {
        let r_c: f64 = params[0];
        let r_lambda: f64 = params[1];

        let r_l1 = weights.abs().sum(Kind::Float) * r_lambda;
        let r_l2 = weights.pow(&Tensor::from(2.0)).sum(Kind::Float) * r_lambda;

        let regularized = match operation {
            RegType::L1 => r_c * r_l1,
            RegType::L2 => r_c * r_l2,
            RegType::Elasticnet => {
                let r_elasticnet = r_c * (r_lambda * r_l1 + (1.0 - r_lambda) * r_l2);
                Tensor::from(r_elasticnet)
            }
        };
        regularized
    }
}



