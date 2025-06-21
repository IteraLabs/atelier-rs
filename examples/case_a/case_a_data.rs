use atelier_data::{data, templates};
use atelier_dcml::{features, targets};
use atelier_synth::synthbooks::progressions;
use std::{env, path::Path};

#[tokio::main]
pub async fn main() {
    // --- Setup working directory
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("Failed to get workspace root");

    // --- Template file (toml)
    let template_file = workspace_root
        .join("examples")
        .join("case_a")
        .join("config_a.toml");

    let template = templates::Config::load_from_toml(template_file.to_str().unwrap())
        .unwrap()
        .clone();

    // --- Extract parameters from template
    let exp_id = &template.experiments[0].id;
    let n_progres = template.experiments[0].n_progressions as usize;
    let template_orderbook = template.exchanges[0].orderbook.clone().unwrap();
    let returns_model = template.models[0].clone();

    // --- Create Orderbook Progressions
    let orderbook = progressions(template_orderbook, returns_model, n_progres);

    // --- Orderbook data file (json)
    let file_name_ob = exp_id.to_owned() + "_ob" + ".json";
    let _folder_route_ob = workspace_root
        .join("examples")
        .join("case_a")
        .join(file_name_ob)
        .to_str()
        .unwrap()
        .to_owned();

    // data::write_to_json(orderbook.as_ref().unwrap(), &folder_route_ob);

    // --- Compute Features from Orderbook Synthetic Data
    let selected_features = ["spread", "midprice", "w_midprice", "vwap", "imb", "tav"];
    let depth: usize = 10;
    let bps: f64 = 1.0;
    let features_vec = features::compute_features(
        &orderbook.as_ref().unwrap(),
        &selected_features,
        depth,
        bps,
        features::FeaturesOutput::Values,
    );

    // println!("features_vec: {:?}", features_vec);

    // -- Compute Target from Orderbook Synthetic Data
    let selected_target = ["return_sign"];
    let target_vec = targets::compute_targets(
        &orderbook.as_ref().unwrap(),
        &selected_target,
        targets::TargetsOutput::Values,
    );

    // println!("target_vec {:?}", target_vec);

    // --- Merge Features and Target

    let pre_dataset = data::Dataset::new()
        .features(features_vec.unwrap().clone())
        .target(target_vec.unwrap().clone())
        .build()
        .unwrap();

    // println!("index: {:?}, features: {:?}, target: {:?}",
    //      dataset.index[0], dataset.features[0], dataset.target[0]);

    let dataset = pre_dataset.shift_features();

    // --- Features and Target file (csv)
    let file_name_ft = exp_id.to_owned() + "_data.csv";
    let features_target_csv = workspace_root
        .join("examples")
        .join("case_a")
        .join(file_name_ft)
        .to_str()
        .unwrap()
        .to_owned();

    data::write_to_csv(&dataset, &features_target_csv);
}
