use serde::Deserialize;
use std::{error::Error, fs};
use toml;

#[derive(Debug, Deserialize, Clone)]
pub enum Models {
    Uniform,
    GBM,
    Hawkes,
    GD,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub experiments: Vec<ExpConfig>,
    pub exchanges: Vec<ExchangeConfig>,
    pub models: Vec<ModelConfig>,
}

impl Config {
    pub fn load_from_toml(file_route: &str) -> Result<Self, Box<dyn Error>> {
        let contents = fs::read_to_string(file_route)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExpConfig {
    pub id: String,
    pub n_progressions: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExchangeConfig {
    pub id: String,
    pub region: String,
    pub name: String,
    pub category: String,
    pub orderbook: Option<OrderbookConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub id: Option<String>,
    pub label: Option<Models>,
    pub description: Option<String>,
    pub params_labels: Option<Vec<String>>,
    pub params_values: Option<Vec<f64>>,
}

impl ModelConfig {
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfigBuilder {
    pub id: Option<String>,
    pub label: Option<Models>,
    pub description: Option<String>,
    pub params_labels: Option<Vec<String>>,
    pub params_values: Option<Vec<f64>>,
}

impl ModelConfigBuilder {
    pub fn new() -> Self {
        ModelConfigBuilder {
            id: None,
            label: None,
            description: None,
            params_labels: None,
            params_values: None,
        }
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    pub fn label(mut self, label: Models) -> Self {
        self.label = Some(label);
        self
    }

    pub fn description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    pub fn params_labels(mut self, params_labels: Vec<String>) -> Self {
        self.params_labels = Some(params_labels);
        self
    }

    pub fn params_values(mut self, params_values: Vec<f64>) -> Self {
        self.params_values = Some(params_values);
        self
    }

    pub fn build(self) -> Result<ModelConfig, &'static str> {
        let id = self.id.ok_or("Missing Model's id")?;
        let label = self.label.ok_or("Missing Model's label")?;
        let description = self.description.ok_or("Missing Model's description")?;
        let params_labels = self.params_labels.ok_or("Missing Model's params_labels")?;
        let params_values = self.params_values.ok_or("Missing Model's params_values")?;

        Ok(ModelConfig {
            id: Some(id),
            label: Some(label),
            description: Some(description),
            params_labels: Some(params_labels),
            params_values: Some(params_values),
        })
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct OrderbookConfig {
    pub update_freq: Option<u64>,
    pub bid_price: Option<f64>,
    pub bid_levels: Option<Vec<u32>>,
    pub bid_orders: Option<Vec<u32>>,
    pub ticksize: Option<Vec<f64>>,
    pub ask_price: Option<f64>,
    pub ask_levels: Option<Vec<u32>>,
    pub ask_orders: Option<Vec<u32>>,
    pub rands: Option<Vec<f64>>,
}

impl OrderbookConfig {
    pub fn builder() -> OrderbookConfigBuilder {
        OrderbookConfigBuilder::new()
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct OrderbookConfigBuilder {
    pub update_freq: Option<u64>,
    pub bid_price: Option<f64>,
    pub bid_levels: Option<Vec<u32>>,
    pub bid_orders: Option<Vec<u32>>,
    pub ticksize: Option<Vec<f64>>,
    pub ask_price: Option<f64>,
    pub ask_levels: Option<Vec<u32>>,
    pub ask_orders: Option<Vec<u32>>,
    pub rands: Option<Vec<f64>>,
}

impl OrderbookConfigBuilder {
    pub fn new() -> Self {
        OrderbookConfigBuilder {
            update_freq: None,
            bid_price: None,
            bid_levels: None,
            bid_orders: None,
            ticksize: None,
            ask_price: None,
            ask_levels: None,
            ask_orders: None,
            rands: None,
        }
    }

    pub fn update_freq(mut self, update_freq: u64) -> Self {
        self.update_freq = Some(update_freq);
        self
    }

    pub fn bid_price(mut self, bid_price: f64) -> Self {
        self.bid_price = Some(bid_price);
        self
    }

    pub fn bid_levels(mut self, bid_levels: Vec<u32>) -> Self {
        self.bid_levels = Some(bid_levels);
        self
    }

    pub fn bid_orders(mut self, bid_orders: Vec<u32>) -> Self {
        self.bid_orders = Some(bid_orders);
        self
    }

    pub fn ask_price(mut self, ask_price: f64) -> Self {
        self.ask_price = Some(ask_price);
        self
    }

    pub fn ask_levels(mut self, ask_levels: Vec<u32>) -> Self {
        self.ask_levels = Some(ask_levels);
        self
    }

    pub fn ask_orders(mut self, ask_orders: Vec<u32>) -> Self {
        self.ask_orders = Some(ask_orders);
        self
    }

    pub fn ticksize(mut self, ticksize: Vec<f64>) -> Self {
        self.ticksize = Some(ticksize);
        self
    }

    pub fn build(self) -> Result<OrderbookConfig, &'static str> {
        let update_freq = self.update_freq.ok_or("Missing initial update freq")?;
        let bid_price = self.bid_price.ok_or("Missing initial bid price")?;
        let bid_levels = self.bid_levels.ok_or("Missing initial bid levels")?;
        let bid_orders = self.bid_orders.ok_or("Missing initial bid orders")?;
        let ticksize = self.ticksize.ok_or("Missing initial tick size")?;
        let ask_price = self.ask_price.ok_or("Missing initial ask price")?;
        let ask_levels = self.ask_levels.ok_or("Missing initial ask levels")?;
        let ask_orders = self.ask_orders.ok_or("Missing initial ask orders")?;
        let rands = self.rands.ok_or("Missing initial random numbers")?;

        Ok(OrderbookConfig {
            update_freq: Some(update_freq),
            bid_price: Some(bid_price),
            bid_levels: Some(bid_levels),
            bid_orders: Some(bid_orders),
            ticksize: Some(ticksize),
            ask_price: Some(ask_price),
            ask_levels: Some(ask_levels),
            ask_orders: Some(ask_orders),
            rands: Some(rands),
        })
    }
}
