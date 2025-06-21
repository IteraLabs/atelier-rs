//! # Brownian Motion
//!
//! ## Geometric
//!
//! The dynamics of GBM can be described by the following
//! Stochastic Differential Equation:
//!
//! $$
//!  dS_{t} = \mu S_{t} dt + \sigma S_{t} dW_{t}
//! $$
//!
//! #### Where:
//!
//! $S_{t}$: The price of the asset at time $t$. \
//! $\mu$: The drift coefficient, representing the expected return. \
//! $\sigma$: The diffusion coefficient, representing the degree of variation in
//! returns. \ $dW_{t}$: A $Wiener$ process as the source of randomness within
//! the model. \
//!
//! ## Implementation
//! $\mu$: Is sometimes used to represent a $trend$ in the prices. \
//! $\sigma$: Is sometimes used to represent the $volatility$ of the asset's
//! return. \ $dS_{t}$: is a non-negative, log-normally distributed, i.i.d
//! process. Frequently used for Pricing Options (Black-Scholes), Value-at-risk
//! (VaR) calculations, dummy examples of prices progression.
//!
//! ## References
//! [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion)

use crate::{probabilistic, probabilistic::Sampling};
use atelier_results::errors::GeneratorError;

fn gbm_return_valid_inputs(
    s0: &f64,
    sigma: &f64,
    dt: &f64,
    n: &usize,
) -> Result<(), GeneratorError> {
    match dt.is_sign_positive() && sigma >= &0.0 && s0 >= &0.0 && dt > &0.0 && n > &0 {
        true => Ok(()),
        false => Err(GeneratorError::GeneratorInputTypeFailure),
    }
}

pub fn gbm_return(
    s0: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    n: usize,
) -> Result<Vec<f64>, GeneratorError> {

    match gbm_return_valid_inputs(&s0, &sigma, &dt, &n) {
        Ok(()) => {
            let dis = probabilistic::NormalDistribution {
                mu: 0.0,
                sigma: dt.sqrt(),
            };

            if n == 1 {
                
                let dwt = dt.sqrt() * dis.sample(n as usize)[0];
                let drift = mu * s0 * dt;
                let diffusion = sigma * s0 * dwt;
                let dst = drift + diffusion;

                Ok(vec![dst])

            } else {

                let dwt: Vec<f64> = dis.sample(n).clone().into_iter().collect();
                let mut v_ds = vec![];
                let mut v_s = vec![s0];

                // progress through all the future steps
                for t in 0..dwt.len() {
                    // drift = mu * s_t * d_t
                    let drift = mu * v_s[t] * dt;
                    // diffusion = sigma * s_t * dW_t
                    let diffusion = sigma * v_s[t] * dwt[t];
                    // dS_t = drift_t + diffusion_t
                    let ds = drift + diffusion;
                    // Update vector of prices
                    v_s.push(v_s[t] + ds);
                    // Update vector of differences of prices
                    v_ds.push(ds);
                }

                Ok(v_ds)

            }
        }
        Err(e) => Err(e),
    }
}

