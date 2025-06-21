use tch::{no_grad, Tensor};

pub trait DistributedOptimizer {
    fn id(&mut self, id: String);
    fn step(
        &self, 
        v_weights: Vec<&mut Tensor>,
        v_weights_gradients : Vec<&Tensor>,
        topology: &Tensor,
        consensus: &Tensor,
    );
}

#[derive(Debug)]
pub struct DGD {
    id: String,
    learning_rate: f64,
}

impl DGD {
    pub fn new() -> OptimizerBuilder {
        OptimizerBuilder::new()
    }
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

    pub fn build(self) -> Result<DGD, &'static str> {
        let id = self.id.ok_or("Missing id")?;
        let learning_rate = self.learning_rate.ok_or("Missing Learning Rate")?;
        Ok(DGD { id, learning_rate })
    }
}

impl DistributedOptimizer for DGD {

    fn id(&mut self, id: String) {
        self.id = id; 
    }

    fn step(
        &self, 
        mut v_weights: Vec<&mut Tensor>,
        v_weights_gradients : Vec<&Tensor>,
        _topology: &Tensor,
        _consensus: &Tensor,
        ) {

        // compute gradients
        no_grad(|| {
            
            let _ = v_weights[0].f_sub_(&(v_weights_gradients[0] * self.learning_rate));

        })

        // collect current parameters
        // compute consensus
        // compute losses
        // compute metrics
        // update weights
        // update metrics

    }
}

pub struct DistributedOptimizerBuilder {
    id: Option<String>,
    learning_rate: Option<f64>,
}

impl DistributedOptimizerBuilder {
    
    pub fn new() -> Self {
        DistributedOptimizerBuilder {
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

    pub fn build(self) -> Result<DGD, &'static str> {
        let id = self.id.ok_or("Missing id")?;
        let learning_rate = self.learning_rate.ok_or("Missing learning_rate")?;
        Ok(DGD { id, learning_rate })
    }
}


