//! atelier-dcml
//!
//! Distributed Convex Machine Learning for the atelier-rs engine.

/// Singular and Distributed Agents
pub mod agents;

/// Mathematical objects and definitions
pub mod math;

/// Feature engineering
pub mod features;

/// target engineering
pub mod targets;

/// Loss function engineering
pub mod functions;

/// Singular and Distributed Training Processes
pub mod processes;

/// Convex Linear Models
pub mod models;

/// Optimizers and Learning Algorithms
pub mod optimizers;

/// Various metrics
pub mod metrics;

pub mod distributed;
