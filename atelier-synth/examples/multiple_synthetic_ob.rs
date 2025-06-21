use atelier_data::templates;
use atelier_synth::synthbooks::async_progressions;
use std::{error::Error, fs::File, io::Write, path::Path};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // --- USAGE PARAMETERS
    let template_file = "multi_orderbooks.toml";

    // --- Setup working directory
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("Failed to get workspace root");

    // --- Template file (toml)
    let template_file = workspace_root
        .join("atelier-synth")
        .join("templates")
        .join(&template_file);
    let template = templates::Config::load_from_toml(template_file.to_str().unwrap())
        .unwrap()
        .clone();

    // --- Extract parameters from template
    let n_progres = template.experiments[0].n_progressions as usize;
    let v_template_model = template.models;
    let v_template_exchanges = template.exchanges.clone();
    let v_template_orderbook = template
        .exchanges
        .into_iter()
        .map(|exchange| exchange.orderbook.unwrap())
        .collect();

    // --- Execute Orderbook Progressions
    let v_rand_ob =
        async_progressions(v_template_orderbook, v_template_model, n_progres).await;

    // --- Create Orderbook data files

    match v_rand_ob {
        Ok(all_orderbooks) => {
            println!(
                "all {:?} of orderbooks successfully generated",
                &all_orderbooks.len()
            );

            for i in 0..v_template_exchanges.len() {
                // --- Write data into JSON
                let id = &v_template_exchanges[i].id.clone();
                let name = &v_template_exchanges[i].name.clone();

                let mut data_file = workspace_root
                    .join("atelier-synth")
                    .join("datasets")
                    .to_str()
                    .unwrap()
                    .to_owned();

                data_file.push_str("/");
                data_file.push_str(&id);
                data_file.push_str("_");
                data_file.push_str(&name);
                data_file.push_str(".json");

                println!("data_file: {:?}", data_file);

                let ob_json = serde_json::to_string(&all_orderbooks[i]).unwrap();
                let mut file = File::create(&data_file).unwrap();
                file.write_all(ob_json.as_bytes()).unwrap();
            }
        }

        Err(e) => {
            eprintln!("At least one progression failed: {}", e);
        }
    }
    Ok(())
}
