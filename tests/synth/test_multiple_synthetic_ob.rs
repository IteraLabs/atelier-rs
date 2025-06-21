#[cfg(test)]

// -- ----------------------------------------------------------------- TESTS UTILS -- //
// -- ----------------------------------------------------------------- ----------- -- //

mod test_utils {}

mod tests {

    use atelier_data::templates;
    use atelier_synth::synthbooks::async_progressions;
    use std::path::Path;

    // ------------------------------------------------------------- TEST ORDERBOOK -- //

    #[tokio::test]
    async fn test_multiple_synthetic_ob() {
        // --- USAGE PARAMETERS
        let template_file = "multi_orderbooks.toml".to_string();

        // --- Setup working directory
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = Path::new(manifest_dir)
            .parent()
            .expect("Failed to get workspace root");

        // --- Template file (toml)
        let template_content = workspace_root
            .join("atelier-synth")
            .join("templates")
            .join(&template_file);

        let template =
            templates::Config::load_from_toml(template_content.to_str().unwrap())
                .unwrap()
                .clone();

        // --- Extract parameters from template
        let n_progres = template.experiments[0].n_progressions as usize;
        let v_template_model = template.models;
        let n_exchanges = template.exchanges.len();
        let v_template_orderbook = template
            .exchanges
            .into_iter()
            .map(|exchange| exchange.orderbook.unwrap())
            .collect();

        // --- Execute Orderbook Progressions
        let v_rand_ob =
            async_progressions(v_template_orderbook, v_template_model, n_progres).await;

        // --- Create Orderbook data files

        println!("\n");

        match v_rand_ob {
            Ok(all_orderbooks) => {
                println!(
                    "all {:?} of orderbooks successfully asynchronously generated",
                    &all_orderbooks.len()
                );
                let n_obs = all_orderbooks.len();
                assert_eq!(n_obs, n_exchanges);
            }

            Err(e) => {
                eprintln!("At least one progression failed: {}", e);
                assert_eq!(true, false);
            }
        }
    }
}
