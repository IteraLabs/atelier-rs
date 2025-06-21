use atelier_data::templates;
use atelier_dcml::math::Stats;

use clap::Parser;

use atelier_synth::synthbooks;
use std::{
    env,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

#[allow(unreachable_code)]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Template File Path
    #[arg(short, long)]
    template: PathBuf,

    /// Output Directory
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Experiment Suffix (Optional)
    #[arg(short, long, default_value = "ob")]
    suffix: String,
}

#[tokio::main]
pub async fn main() {
    let args = Args::parse();

    // --- Setup working directory
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .expect("Failed to get workspace root");

    // --- Template file (from command line argument)
    let template = templates::Config::load_from_toml(args.template.to_str().unwrap())
        .unwrap()
        .clone();

    // --- Extract parameters from template
    let exp_id = &template.experiments[0].id;
    let n_progres = template.experiments[0].n_progressions as usize;
    let template_orderbook = template.exchanges[0].orderbook.clone().unwrap();
    let template_model = template.models[0].clone();

    // --- Create progressions
    let v_rand_ob = synthbooks::progressions(
        template_orderbook,
        template_model,
        n_progres);

    // --- Compute basic stats
    let level_bids: Vec<f32> = v_rand_ob
        .as_ref()
        .unwrap()
        .iter()
        .map(|x| x.bids.len() as f32)
        .collect();

    let level_asks: Vec<f32> = v_rand_ob
        .as_ref()
        .unwrap()
        .iter()
        .map(|x| x.asks.len() as f32)
        .collect();

    let bids_stats = Stats::new(&level_bids);
    let asks_stats = Stats::new(&level_asks);

    println!("bids_stats: {:?}, asks_stats {:?}", bids_stats, asks_stats);

    // --- Use command line arguments for output
    let mut folder_route = workspace_root
        .join(&args.output_dir)
        .to_str()
        .unwrap()
        .to_owned();

    folder_route.push_str("/");
    folder_route.push_str(&exp_id);
    folder_route.push_str("_");
    folder_route.push_str(&args.suffix);
    folder_route.push_str(".json");

    let i_ob = v_rand_ob.as_ref().unwrap();

    let ob_json = serde_json::to_string(&i_ob).unwrap();
    let mut file = File::create(&folder_route).unwrap();
    file.write_all(ob_json.as_bytes()).unwrap();

    println!("Output written to: {}", folder_route);
}
