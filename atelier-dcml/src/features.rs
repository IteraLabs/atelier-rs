/// Features Calculation
use atelier_data::{data, orderbooks::Orderbook};
use std::error::Error;

#[derive(Debug, Clone, Copy)]
pub enum FeaturesOutput {
    Values,
    HashMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderbookFeatures {
    Spread,
    Midprice,
    WeightedMidprice,
    VWAP,
    Imb,
    TAV,
}

impl OrderbookFeatures {
    pub fn name(&self) -> &'static str {
        match self {
            OrderbookFeatures::Spread => "spread",
            OrderbookFeatures::Midprice => "midprice",
            OrderbookFeatures::WeightedMidprice => "w_midprice",
            OrderbookFeatures::VWAP => "vwap",
            OrderbookFeatures::Imb => "imb",
            OrderbookFeatures::TAV => "tav",
        }
    }

    pub fn compute(
        &self,
        ob: &Orderbook,
        depth: usize,
        bps: f64,
    ) -> Result<f64, Box<(dyn Error + 'static)>> {
        match self {
            OrderbookFeatures::Spread => Ok(compute_spread(ob)),
            OrderbookFeatures::Midprice => Ok(compute_midprice(ob)),
            OrderbookFeatures::WeightedMidprice => Ok(compute_w_midprice(ob)),
            OrderbookFeatures::VWAP => Ok(compute_vwap(ob, depth)),
            OrderbookFeatures::Imb => Ok(compute_imb(ob)),
            OrderbookFeatures::TAV => Ok(compute_tav(ob, bps)),
        }
    }

    pub fn from_name(name: &str) -> Option<OrderbookFeatures> {
        match name {
            "spread" => Some(OrderbookFeatures::Spread),
            "midprice" => Some(OrderbookFeatures::Midprice),
            "w_midprice" => Some(OrderbookFeatures::WeightedMidprice),
            "vwap" => Some(OrderbookFeatures::VWAP),
            "imb" => Some(OrderbookFeatures::Imb),
            "tav" => Some(OrderbookFeatures::TAV),
            _ => None,
        }
    }

    pub fn all_features() -> Vec<OrderbookFeatures> {
        vec![
            OrderbookFeatures::Spread,
            OrderbookFeatures::Midprice,
            OrderbookFeatures::WeightedMidprice,
            OrderbookFeatures::VWAP,
            OrderbookFeatures::Imb,
            OrderbookFeatures::TAV,
        ]
    }

    pub fn list_features() -> Vec<&'static str> {
        Self::all_features().iter().map(|f| f.name()).collect()
    }
}

#[derive(Debug)]
pub struct FeatureSelector {
    selected_features: Vec<OrderbookFeatures>,
}

impl FeatureSelector {
    pub fn new(features_names: &[&str]) -> Result<Self, String> {
        let mut features = Vec::new();
        for name in features_names {
            match OrderbookFeatures::from_name(name) {
                Some(feature) => features.push(feature),
                None => {
                    return Err(format!(
                        "Unknown feature: {}, the ones available are: {:?}",
                        name,
                        OrderbookFeatures::list_features()
                    ));
                }
            }
        }
        Ok(FeatureSelector {
            selected_features: features,
        })
    }

    /// Select Features by Enum Variant
    pub fn from_features(features: Vec<OrderbookFeatures>) -> Self {
        FeatureSelector {
            selected_features: features,
        }
    }

    /// Compute all values
    pub fn compute_values(&self, ob: &Orderbook, depth: usize, bps: f64) -> Vec<f64> {
        self.selected_features
            .iter()
            .map(|feature| feature.compute(ob, depth, bps).unwrap())
            .collect()
    }

    /// Get all features names
    pub fn features_names(&self) -> Vec<&'static str> {
        self.selected_features.iter().map(|f| f.name()).collect()
    }
}

pub fn compute_features(
    orderbooks: &[Orderbook],
    feature_names: &[&str],
    depth: usize,
    bps: f64,
    output_format: FeaturesOutput,
) -> Result<Vec<Vec<f64>>, String> {
    let selector = FeatureSelector::new(feature_names)?;
    let mut feature_matrix = Vec::new();

    for ob in orderbooks {
        let features = selector.compute_values(ob, depth, bps);
        feature_matrix.push(features);
    }

    match output_format {
        FeaturesOutput::Values => Ok(feature_matrix),
        FeaturesOutput::HashMap => Ok(feature_matrix),
    }
}

// --- Different Features Computations --- //

/// Spread
pub fn compute_spread(ob: &Orderbook) -> f64 {
    let i_spread = &ob.asks[0].price - &ob.bids[0].price;
    data::truncate_to_decimal(i_spread, 8)
}

/// Midprice
pub fn compute_midprice(ob: &Orderbook) -> f64 {
    let i_midprice = (&ob.asks[0].price + &ob.bids[0].price) / 2.0;
    data::truncate_to_decimal(i_midprice, 8)
}

/// Weighted Midprice
pub fn compute_w_midprice(ob: &Orderbook) -> f64 {
    let i_w_midprice = ((ob.bids[0].price * ob.bids[0].volume)
        + (ob.asks[0].price * ob.asks[0].volume))
        / (ob.asks[0].volume + ob.bids[0].volume);
    data::truncate_to_decimal(i_w_midprice, 8)
}

/// Orderbook Volume Imbalance
pub fn compute_imb(ob: &Orderbook) -> f64 {
    let i_imb = &ob.asks[0].volume / (&ob.asks[0].volume + &ob.bids[0].volume);
    data::truncate_to_decimal(i_imb, 8)
}

/// Volume-Weighted Average Price (With depth selector)
///
/// Takes the orderbook bids and asks, up to the specified level, and
/// calculates the classic Volume-Weighted Average Price.
pub fn compute_vwap(ob: &Orderbook, depth: usize) -> f64 {
    let bid_levels = ob.bids.iter().take(depth);
    let ask_levels = ob.asks.iter().take(depth);
    let all_levels = bid_levels.chain(ask_levels);

    let (sum_p_v, sum_v) = all_levels.fold((0.0, 0.0), |(acc_p_v, acc_v), level| {
        (acc_p_v + level.price * level.volume, acc_v + level.volume)
    });
    if sum_v > 0.0 {
        let vwap = sum_p_v / sum_v;
        data::truncate_to_decimal(vwap, 8)
    } else {
        0.0
    }
}

/// Total Available Volume
///
/// The total volume posted in the orderbook within X bps of the midprice
///
pub fn compute_tav(ob: &Orderbook, bps: f64) -> f64 {
    let best_bid = &ob.bids[0].price;
    let best_ask = &ob.asks[0].price;
    let upper_ask = best_ask * (1.0 + bps);
    let lower_bid = best_bid * (1.0 - bps);

    // find the closest bid leve to lower bid
    let bid_volume: f64 = ob
        .bids
        .iter()
        .filter(|level| level.price >= lower_bid)
        .map(|level| level.volume)
        .sum();

    let ask_volume: f64 = ob
        .asks
        .iter()
        .filter(|level| level.price <= upper_ask)
        .map(|level| level.volume)
        .sum();

    let i_tav = bid_volume + ask_volume;
    data::truncate_to_decimal(i_tav, 8)
}

/// Orderbook Updates Distribution
///
/// From the timeseries of timestamps where a new orderbook was formed
///
pub fn compute_obts(obs: &Vec<Orderbook>) -> Vec<f64> {

    let ob_ts = obs
        .iter()
        .map(|ob| ob.orderbook_ts.clone() as f64)
        .collect::<Vec<_>>()
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();

    ob_ts

}

