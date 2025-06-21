use atelier_data::{
    orderbooks::Orderbook,
    templates::{ModelConfig, Models, OrderbookConfig},
};
use atelier_results::errors;
use atelier_generators::{brownian, probabilistic};
use tokio::task;
use futures::future::join_all;
use std::error::Error;

const SECONDS_IN_YEAR: u32  =  31_557_600;
const MIN_SPREAD: f64 = 0.001; 

/// Generates a randomized orderbook snapshot based on input parameters.
///
/// Constructs an orderbook using the Orderbook::random to model price evolution,
/// with configurable market depth and order characteristics.
///
/// # Arguments
/// - `bid_price`: Initial best bid price
/// - `bid_levels`: [min, max] number of price levels for bids
/// - `bid_orders`: [min, max] number of orders per bid level
/// - `ticksize`: [min, max] price increment between levels
/// - `ask_price`: Initial best ask price
/// - `ask_levels`: [min, max] number of price levels for asks
/// - `ask_orders`: [min, max] number of orders per ask level
///
/// # Returns
/// `Result<Orderbook>` containing either:
/// - Randomized orderbook snapshot
/// - Error if input validation fails
///
/// # Panics
/// - If any vector argument doesn't contain exactly 2 elements
/// - If bid_price >= ask_price (violates market structure)
///
pub fn progress(
    update_ts: u64,
    bid_price: f64,
    bid_levels: Vec<u32>,
    bid_orders: Vec<u32>,
    ticksize: Vec<f64>,
    ask_price: f64,
    ask_levels: Vec<u32>,
    ask_orders: Vec<u32>,
) -> Result<Orderbook, Box<dyn Error>> {
    let r_ob = Orderbook::random(
        Some(update_ts),
        bid_price,
        Some((bid_levels[0], bid_levels[1])),
        Some((bid_orders[0], bid_orders[1])),
        Some((ticksize[0], ticksize[1])),
        ask_price,
        Some((ask_levels[0], ask_levels[1])),
        Some((ask_orders[0], ask_orders[1])),
    );

    Ok(r_ob)
}

/// Generates a sequence of orderbook progressions using Brownian motion dynamics.
///
/// This async function creates a time series of orderbooks where each subsequent book:
/// - Inherits structure from previous state
/// - Evolves prices using GBM returns
/// - Maintains configurable market depth parameters
///
/// # Arguments
/// - `template_orderbook`: Initial configuration with all fields required
/// - `template_model`: GBM parameters (μ, σ) required
/// - `n_progres`: Number of progressions to generate
///
/// # Returns
/// `Result<Vec<Orderbook>>` containing either:
/// - Time series of orderbook states
/// - Error if input validation fails or model becomes unstable
///
/// # Panics
/// - If any template field contains `None`
/// - If μ or σ lead to negative prices
///
pub fn progressions(
    template_orderbook: OrderbookConfig,
    template_model: ModelConfig,
    n_progres: usize,
) -> Result<Vec<Orderbook>, errors::SynthetizerError> {
    let mut v_orderbooks: Vec<Orderbook> = Vec::new();

    // Extract and calculate initial values
    let update_freq = template_orderbook.update_freq.unwrap() as f64;
    let ini_bid = template_orderbook.bid_price.unwrap();
    let ini_ask = template_orderbook.ask_price.unwrap();
    

    // Initialize current prices for cumulative evolution
    let mut current_bid_price = ini_bid;
    let mut current_ask_price = ini_ask;

    let model_label: Models = template_model.clone().label.unwrap();

    for _i in 0..n_progres {

        let (bid_return, ask_return) = match model_label {

            Models::Uniform => {
                let lower = template_model.clone().params_values.as_ref().unwrap()[0];
                let upper = template_model.clone().params_values.as_ref().unwrap()[1];

                let bid_return = probabilistic::uniform_return(lower, upper, 1)[0];
                let ask_return = probabilistic::uniform_return(lower, upper, 1)[0];

                (bid_return, ask_return)
            }

            Models::GBM => {

                let mu = template_model.clone().params_values.as_ref().unwrap()[0];
                let sigma = template_model.clone().params_values.unwrap()[1];
                let dt = (update_freq/1e3) / SECONDS_IN_YEAR as f64 ;
                let n = 1;

                let bid_return = brownian::gbm_return(
                    current_bid_price,
                    mu,
                    sigma,
                    dt,
                    n
                ).unwrap()[0];

                let ask_return = brownian::gbm_return(
                    current_ask_price,
                    mu,
                    sigma,
                    dt,
                    n
                ).unwrap()[0];

                (bid_return, ask_return)

            }

            _ => (0.001, 0.001),

        };
        

        let delta_bid = current_bid_price * bid_return;
        let delta_ask = current_ask_price * ask_return;

        // Apply deltas while maintaining spread integrity
        let mut new_bid = current_bid_price + delta_bid;
        let mut new_ask = current_ask_price + delta_ask;

        // Enforce spread constraints
        if new_bid >= new_ask {

            // Calculate midprice and reset spread
            let mid = (new_bid + new_ask) / 2.0;
            new_bid = mid - MIN_SPREAD / 2.0;
            new_ask = mid + MIN_SPREAD / 2.0;
        
        } else if (new_ask - new_bid) < MIN_SPREAD {

            // Widen spread to minimum without moving midprice
            let adjustment = (MIN_SPREAD - (new_ask - new_bid)) / 2.0;
            new_bid -= adjustment;
            new_ask += adjustment;

        }

        // Apply validated prices
        current_bid_price = new_bid;
        current_ask_price = new_ask;
      
        let bid_levels = template_orderbook.clone().bid_levels.unwrap();
        let bid_orders = template_orderbook.clone().bid_orders.unwrap();
        let ticksize = template_orderbook.clone().ticksize.unwrap();
        let ask_levels = template_orderbook.clone().ask_levels.unwrap();
        let ask_orders = template_orderbook.clone().ask_orders.unwrap();

        let r_ob = Orderbook::random(
            None,
            current_bid_price,
            Some((bid_levels[0], bid_levels[1])),
            Some((bid_orders[0], bid_orders[1])),
            Some((ticksize[0], ticksize[1])),
            current_ask_price,
            Some((ask_levels[0], ask_levels[1])),
            Some((ask_orders[0], ask_orders[1])),
        );

        // --- Progress next Orderbook
        current_bid_price = r_ob.bids[0].price.clone();
        current_ask_price = r_ob.asks[0].price.clone();

        v_orderbooks.push(r_ob);

    }
 
    Ok(v_orderbooks)
}

/// Executes multiple orderbook progression scenarios concurrently.
///
/// This high-performance implementation uses async rust to parallelize:
/// - Different initial orderbook configurations
/// - Multiple model parameterizations
/// - Independent progression sequences
///
/// # Arguments
/// - `orderbooks`: Vector of unique initial orderbook states
/// - `models`: Corresponding vector of model configurations
/// - `n_progres`: Number of steps per progression sequence
///
/// # Returns
/// Vector of individual progression results, preserving input order
///
/// # Note
/// Each progression task fails independently - check individual results
/// for partial successes in distributed computing scenarios
///
pub async fn async_progressions(

    orderbook_templates: Vec<OrderbookConfig>,
    model_templates: Vec<ModelConfig>,
    n_progres: usize,
    ) -> Result<Vec<Vec<Orderbook>>, errors::SynthetizerError> {
    
    let tasks: Vec<_> = orderbook_templates
    .into_iter()
    .zip(model_templates.into_iter())
    .map(|(ob, model)| {
        async move {
                task::spawn_blocking(move || progressions(ob, model, n_progres))
                .await
                .map_err(|e| errors::SynthetizerError::GenerationError(e.to_string()))
            }
        })
        .collect();

    let results = join_all(tasks).await;

    // Process the results
    // Process results: aggregate successes or return first error
    let mut all_progressions = Vec::with_capacity(results.len());

    println!("results: {:?}", results.len());

    for res in results {

        match res {
            
            // Handle task join error
            Err(join_err) => {
                errors::SynthetizerError::GenerationError(join_err.to_string());
            },

            // Handle progression execution error
            Ok(Err(e)) => return Err(e),

            // Aggregate successful orderbooks
            Ok(Ok(v_orderbook)) => all_progressions.push(v_orderbook),
        }
    }

    Ok(all_progressions)

}

