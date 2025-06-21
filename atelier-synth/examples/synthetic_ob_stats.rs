//! Generation of Synthetic Orderbook
//!
//! This code is to generate 3 different markets (of the same pair)
//! with slightly different values in order to explore
//! distributed convex optimization theory
//!

use atelier_data::{orderbooks::Orderbook, templates};
use rand::Rng;
use rand_distr::Uniform;
use serde_json;
use std::{env, fs::File, io::Write, path::Path};

// ----------------------------------------------------------------------------------- //
fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("Failed to get workspace root");

    // Config file (toml)
    let config_file = workspace_root
        .join("atelier-synth")
        .join("templates")
        .join("single_orderbook.toml");

    // Load configuration
    let config = templates::Config::load_from_toml(config_file.to_str().unwrap())
        .unwrap()
        .clone();

    let exchanges = config.exchanges.clone();

    // Number of progressions for all Orderbooks
    let n_progressions = config.experiments[0].n_progressions;
    let exp_id = &config.experiments[0].id;

    for i_e in 0..exchanges.len() {
        // --- File Name Formation Orderbook -- //
        let mut folder_route = workspace_root
            .join("atelier-dcm")
            .join("datasets")
            .to_str()
            .unwrap()
            .to_owned();

        let e_name = config.exchanges[i_e].clone().name.to_owned();
        let e_id = config.exchanges[i_e].clone().id.to_owned();

        // -- Orderbooks vector
        let mut v_orderbook = vec![];

        // --- Generation Parameters
        let bid_price = exchanges[i_e].orderbook.clone().unwrap().bid_price.unwrap();
        let bid_levels = exchanges[i_e]
            .orderbook
            .clone()
            .unwrap()
            .bid_levels
            .unwrap();
        let bid_orders = exchanges[i_e]
            .orderbook
            .clone()
            .unwrap()
            .bid_orders
            .unwrap();
        let ticksize = exchanges[i_e].orderbook.clone().unwrap().ticksize.unwrap();
        let ask_price = exchanges[i_e].orderbook.clone().unwrap().ask_price.unwrap();
        let ask_levels = exchanges[i_e]
            .orderbook
            .clone()
            .unwrap()
            .ask_levels
            .unwrap();
        let ask_orders = exchanges[i_e]
            .orderbook
            .clone()
            .unwrap()
            .ask_orders
            .unwrap();

        // Random numbers
        let uni_params = vec![0.01, 0.11];
        let mut rng = rand::rng();

        // --- Generation progressions
        for _ in 0..n_progressions {
            let uni_rand = Uniform::new(uni_params[0], uni_params[1])
                .expect("Failed to create Uniform distribution sampler");

            let r_amount_ret = rng.sample(uni_rand);

            let v_bid_price = bid_price + bid_price * r_amount_ret;
            let v_ask_price = ask_price + ask_price * r_amount_ret;

            let r_ob = Orderbook::random(
                None,
                v_bid_price,
                Some((bid_levels[0], bid_levels[1])),
                Some((bid_orders[0], bid_orders[1])),
                Some((ticksize[0], ticksize[1])),
                v_ask_price,
                Some((ask_levels[0], ask_levels[1])),
                Some((ask_orders[0], ask_orders[1])),
            );

            v_orderbook.push(r_ob);
        }

        // --- Write into JSON "asia_orderbook.json"
        folder_route.push_str("/");
        folder_route.push_str(&e_id);
        folder_route.push_str(&"_");
        folder_route.push_str(&e_name);
        folder_route.push_str("_ob.json");

        println!("folder_route: {:?}", folder_route);

        //let ob_json = serde_json::to_string(&v_orderbook).unwrap();
        //let mut file = File::create(&folder_route).unwrap();
        //file.write_all(ob_json.as_bytes()).unwrap();

        // --- Presenting results --- //

        println!("\n \n---- {}_{}_ob ---", e_id, e_name);

        println!("\nTOB:");
        println!(
            "
          bid[0].volume {:.4},
          bid[0].price {:.4},
          ask[0].price {:.4},
          ask[0].volume {:.4},
          bid[-1].volume {:.4},
          bid[-1].price {:.4},
          ask[-1].price {:.4},
          ask[-1].volume {:.4}",
            v_orderbook[0].bids[0].volume,
            v_orderbook[0].bids[0].price,
            v_orderbook[0].asks[0].price,
            v_orderbook[0].asks[0].volume,
            v_orderbook[v_orderbook.len() - 1].bids[0].volume,
            v_orderbook[v_orderbook.len() - 1].bids[0].price,
            v_orderbook[v_orderbook.len() - 1].asks[0].price,
            v_orderbook[v_orderbook.len() - 1].asks[0].volume,
        );

        let level_bids: &Vec<f32> =
            &v_orderbook.iter().map(|x| x.bids.len() as f32).collect();
        let (level_bids_min, level_bids_max) = v_orderbook
            .iter()
            .map(|x| x.bids.len() as f32)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), val| {
                (min.min(val), max.max(val))
            });

        let stats_bids = vec![
            level_bids_min,
            level_bids_max,
            level_bids.into_iter().sum::<f32>() / level_bids.into_iter().len() as f32,
        ];

        let level_asks: &Vec<f32> =
            &v_orderbook.iter().map(|x| x.asks.len() as f32).collect();
        let (level_asks_min, level_asks_max) = v_orderbook
            .iter()
            .map(|x| x.asks.len() as f32)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), val| {
                (min.min(val), max.max(val))
            });

        let stats_asks = vec![
            level_asks_min,
            level_asks_max,
            level_asks.into_iter().sum::<f32>() / level_asks.into_iter().len() as f32,
        ];

        println!("\nLEVEL STATS: ");
        println!(
            "\n  No. of Bids levels stats (min, max, mean) {:?}",
            stats_bids
        );
        println!(
            "  No. of Asks levels stats (min, max, mean) {:?}\n",
            stats_asks
        );
    }
}
