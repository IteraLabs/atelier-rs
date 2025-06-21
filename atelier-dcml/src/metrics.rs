use std::collections::HashMap;
use tch::{Kind, Tensor};

// Metric Trait for all metrics
pub trait Metric: std::fmt::Debug {
    fn id(&self) -> &str;
    fn metric_type(&self) -> MetricType;
    fn update(&mut self, value: MetricValue);
    fn latest(&self) -> Option<&MetricValue>;
    fn history(&self) -> &Vec<MetricValue>;
    fn reset(&mut self);
    fn compute(
        &self,
        y_true: &Tensor,
        y_hat: &Tensor,
        threshold: Option<f64>,
    ) -> MetricValue;
}

#[derive(Debug, Clone)]
pub enum MetricType {
    Numerical,
    Categorical,
    Matrix,
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Scalar(f64),
    Matrix(Vec<Vec<f64>>),
    Multiple(HashMap<String, f64>),
}

impl MetricValue {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            MetricValue::Scalar(val) => Some(*val),
            _ => None,
        }
    }

    pub fn as_matrix(&self) -> Option<&Vec<Vec<f64>>> {
        match self {
            MetricValue::Matrix(mat) => Some(mat),
            _ => None,
        }
    }

    pub fn as_multiple(&self) -> Option<&HashMap<String, f64>> {
        match self {
            MetricValue::Multiple(mul) => Some(mul),
            _ => None,
        }
    }
}

// Helper struct for confusion matrix calculations
#[derive(Debug, Clone)]
pub struct ConfusionMatrixComponents {
    pub true_positive: f64,
    pub false_positive: f64,
    pub false_negative: f64,
    pub true_negative: f64,
}

impl ConfusionMatrixComponents {
    pub fn from_tensors(y_true: &Tensor, y_pred: &Tensor, threshold: Option<f64>) -> Self {
        let threshold = threshold.unwrap_or(0.5);
        
        // For classification, use argmax instead of threshold
        let predictions = if y_pred.size().len() > 1 && y_pred.size()[1] > 1 {
            y_pred.argmax(1, false) // Multi-class classification
        } else {
            y_pred.ge(threshold).to_kind(Kind::Float) // Binary classification
        };
        
        let labels = y_true.to_kind(Kind::Float);
        
        // Calculate components correctly
        let _correct = predictions.eq_tensor(&labels);
        let _total_samples = labels.size()[0] as f64;
        
        // For binary classification
        let tp_tensor: Tensor = &predictions * &labels;
        let tp: f64 = tp_tensor.sum(Kind::Float).double_value(&[]);

        let tn_tensor: Tensor = (1.0 - &predictions) * (1.0 - &labels);
        let tn: f64 = tn_tensor.sum(Kind::Float).double_value(&[]);

        let fp_tensor: Tensor = &predictions * (1.0 - &labels);
        let fp: f64 = fp_tensor.sum(Kind::Float).double_value(&[]);

        let fn_tensor: Tensor = (1.0 - &predictions) * &labels;
        let fn_val: f64 = fn_tensor.sum(Kind::Float).double_value(&[]);
        
        ConfusionMatrixComponents {
            true_positive: tp.max(0.0),
            true_negative: tn.max(0.0),
            false_positive: fp.max(0.0),
            false_negative: fn_val.max(0.0),
        }
    }
    // Add this method
    pub fn total(&self) -> f64 {
        self.true_positive + 
        self.true_negative + 
        self.false_positive + 
        self.false_negative
    }

}

// Concrete metric implementations
#[derive(Debug)]
pub struct Accuracy {
    id: String,
    values: Vec<MetricValue>,
}

impl Accuracy {
    pub fn new() -> Self {
        Accuracy {
            id: "accuracy".to_string(),
            values: Vec::new(),
        }
    }
}

impl Metric for Accuracy {
    fn id(&self) -> &str {
        &self.id
    }

    fn metric_type(&self) -> MetricType {
        MetricType::Numerical
    }

    fn compute(&self, y: &Tensor, y_pred: &Tensor, thr: Option<f64>) -> MetricValue {
        let threshold = thr.unwrap_or(0.5);
        let cm = ConfusionMatrixComponents::from_tensors(y, y_pred, Some(threshold));
        let accuracy = (cm.true_positive + cm.true_negative) / cm.total();

        MetricValue::Scalar(accuracy)
    }

    fn update(&mut self, value: MetricValue) {
        self.values.push(value);
    }

    fn latest(&self) -> Option<&MetricValue> {
        self.values.last()
    }

    fn history(&self) -> &Vec<MetricValue> {
        &self.values
    }

    fn reset(&mut self) {
        self.values.clear();
    }
}

#[derive(Debug)]
pub struct ConfusionMatrix {
    id: String,
    values: Vec<MetricValue>,
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        ConfusionMatrix {
            id: "confusion_matrix".to_string(),
            values: Vec::new(),
        }
    }
}

impl Metric for ConfusionMatrix {
    fn id(&self) -> &str {
        &self.id
    }

    fn metric_type(&self) -> MetricType {
        MetricType::Matrix
    }

    fn compute(
        &self,
        y: &Tensor,
        y_pred: &Tensor,
        threshold: Option<f64>,
    ) -> MetricValue {
        let threshold = threshold.unwrap_or(0.5);
        let cm = ConfusionMatrixComponents::from_tensors(y, y_pred, Some(threshold));

        // Return as 2x2 matrix: [[TN, FP], [FN, TP]]
        let matrix = vec![
            vec![cm.true_negative, cm.false_positive],
            vec![cm.false_negative, cm.true_positive],
        ];

        MetricValue::Matrix(matrix)
    }

    fn update(&mut self, value: MetricValue) {
        self.values.push(value);
    }

    fn latest(&self) -> Option<&MetricValue> {
        self.values.last()
    }

    fn history(&self) -> &Vec<MetricValue> {
        &self.values
    }

    fn reset(&mut self) {
        self.values.clear();
    }
}

#[derive(Debug)]
pub struct ClassificationMetrics {
    id: String,
    values: Vec<MetricValue>,
}

impl ClassificationMetrics {
    pub fn new() -> Self {
        ClassificationMetrics {
            id: "classification_metrics".to_string(),
            values: Vec::new(),
        }
    }
}

impl Metric for ClassificationMetrics {
    fn id(&self) -> &str {
        &self.id
    }

    fn metric_type(&self) -> MetricType {
        MetricType::Matrix
    }

    fn compute(
        &self,
        y_true: &Tensor,
        y_pred: &Tensor,
        threshold: Option<f64>,
    ) -> MetricValue {
        let cm = ConfusionMatrixComponents::from_tensors(y_true, y_pred, threshold);

        let accuracy = (cm.true_positive + cm.true_negative) / cm.total();
        let precision = if cm.true_positive + cm.false_positive > 0.0 {
            cm.true_positive / (cm.true_positive + cm.false_positive)
        } else {
            0.0
        };
        let recall = if cm.true_positive + cm.false_negative > 0.0 {
            cm.true_positive / (cm.true_positive + cm.false_negative)
        } else {
            0.0
        };
        let specificity = if cm.true_negative + cm.false_positive > 0.0 {
            cm.true_negative / (cm.true_negative + cm.false_positive)
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let mut results = HashMap::new();

        results.insert("accuracy".to_string(), accuracy);
        results.insert("precision".to_string(), precision);
        results.insert("recall".to_string(), recall);
        results.insert("specificity".to_string(), specificity);
        results.insert("f1_score".to_string(), f1);
        results.insert("tp".to_string(), cm.true_positive);
        results.insert("fp".to_string(), cm.false_positive);
        results.insert("fn".to_string(), cm.false_negative);
        results.insert("tn".to_string(), cm.true_negative);

        MetricValue::Multiple(results)
    }

    fn update(&mut self, value: MetricValue) {
        self.values.push(value);
    }

    fn latest(&self) -> Option<&MetricValue> {
        self.values.last()
    }

    fn history(&self) -> &Vec<MetricValue> {
        &self.values
    }

    fn reset(&mut self) {
        self.values.clear();
    }
}

// Container for multiple metrics
#[derive(Debug)]
pub struct Metrics {
    pub metrics: Vec<Box<dyn Metric>>,
    pub threshold: f64,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            metrics: Vec::new(),
            threshold: 0.5,
        }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Metrics {
            metrics: Vec::new(),
            threshold,
        }
    }

    pub fn add_metric(&mut self, metric: Box<dyn Metric>) {
        self.metrics.push(metric);
    }

    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    pub fn compute_all(
        &mut self,
        y_true: &Tensor,
        y_pred: &Tensor,
    ) -> HashMap<String, MetricValue> {
        let mut results = HashMap::new();

        for metric in self.metrics.iter_mut() {
            let value = metric.compute(y_true, y_pred, Some(self.threshold));
            metric.update(value.clone());
            results.insert(metric.id().to_string(), value);
        }

        results
    }

    pub fn get_latest(&self, metric_id: &str) -> Option<&MetricValue> {
        self.metrics
            .iter()
            .find(|m| m.id() == metric_id)
            .and_then(|m| m.latest())
    }

    pub fn get_history(&self, metric_id: &str) -> Option<&Vec<MetricValue>> {
        self.metrics
            .iter()
            .find(|m| m.id() == metric_id)
            .map(|m| m.history())
    }

    pub fn reset_all(&mut self) {
        for metric in self.metrics.iter_mut() {
            metric.reset();
        }
    }

    pub fn list_metrics(&self) -> Vec<&str> {
        self.metrics.iter().map(|m| m.id()).collect()
    }
}

// Convenience constructors
impl Metrics {
    pub fn complete_classification() -> Self {
        let mut metrics = Metrics::new();
        metrics.add_metric(Box::new(ClassificationMetrics::new()));
        metrics
    }

    pub fn basic_classification() -> Self {
        let mut metrics = Metrics::new();
        metrics.add_metric(Box::new(Accuracy::new()));
        metrics.add_metric(Box::new(ConfusionMatrix::new()));
        metrics
    }
}
