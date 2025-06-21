use atelier_data::templates;
use atelier_synth::synthbooks::progressions;
use std::{env, path::Path};

pub fn main() {

    // --- Setup working directory
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("Failed to get workspace root");

    // --- Template file (toml)
    let template_file = workspace_root
        .join("tests")
        .join("templates")
        .join("test_template_00.toml");

    let template = templates::Config::load_from_toml(
        template_file
            .to_str()
            .unwrap())
        .unwrap()
        .clone();

    // --- Extract parameters from template
    let returns_model = template.models[1].clone();
    println!("model: {:?}", returns_model);

    let n_progres = template.experiments[0].n_progressions as usize;
    println!("n_progres: {:?}", n_progres);

    let template_orderbook = template.exchanges[0].orderbook.clone().unwrap();
    println!("template_orderbook: {:?}", template_orderbook);

    // --- Create progressions
    let orderbooks = progressions(
        template_orderbook,
        returns_model,
        n_progres
    ).unwrap();

    // --- Print results for debug purposes
    for i in 0..n_progres {

        println!("orderbook_{} - bid {:.4} - ask {:.4} - spread {:.4} - mid {:.4}",
            i,
            orderbooks[i].bids[0].price,
            orderbooks[i].asks[0].price,
            orderbooks[i].asks[0].price - orderbooks[i].bids[0].price,
            (orderbooks[i].bids[0].price + orderbooks[i].asks[0].price) / 2.0,
        )
    }

}

