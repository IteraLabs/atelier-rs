use atelier_dcml::{functions, metrics, models, optimizers, processes};

use std::{error::Error, path::Path};
use tch::Device;

use atelier_data::{
    data::{Dataset, Transformation},
    templates,
};

pub fn main() -> Result<(), Box<dyn Error + 'static>> {
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
    let _exp_id = &template.experiments[0].id;
    let n_progres = template.experiments[0].n_progressions as usize;
    let optimizer_model = template.models[1].params_values.clone().unwrap();

    // --- Data Layer --- //

    // File specification and read
    let data_file = workspace_root
        .join("examples")
        .join("case_a")
        .to_str()
        .unwrap()
        .to_owned()
        + "/case_a_data.csv";

    let header = true;
    let column_types = None;
    let target_column = Some(7);

    let mut a_dataset =
        Dataset::from_csv(&data_file, header, column_types, target_column).unwrap();

    a_dataset.transform(Transformation::Scale);

    println!("dataset.features: {:?}", a_dataset.features[0]);

    // --- Model Layer --- //
    let n_inputs = 6;

    let a_model = models::LinearModel::new(n_inputs)
        .id("model_00".to_string())
        .device(Device::Cpu)
        .glorot_uniform_init();

    println!("a_model: {:?}", a_model);

    // --- Optimizer Layer --- //

    let a_optimizer = optimizers::GradientDescent::new()
        .id("opt_00".to_string())
        .learning_rate(optimizer_model[0])
        .build()
        .unwrap();

    // println!("a_optimizer: {:?}", a_optimizer);

    // --- Loss Function Layer --- //

    let a_loss = functions::CrossEntropy::new()
        .id(&"loss_00".to_string())
        .build()
        .unwrap();

    // println!("a_loss: {:?}", a_loss);

    // --- Metrics Layer --- //

    let a_metrics = metrics::Metrics::basic_classification();

    // println!("a_metrics: {:?}", a_metrics.list_metrics());

    // --- Trainer Environment (Singular) --- //

    let mut singular = processes::Singular::new()
        .dataset(a_dataset)
        .model(a_model)
        .loss(a_loss)
        .optimizer(a_optimizer)
        .metrics(a_metrics)
        .build()
        .unwrap();

    let epochs = n_progres;

    let _ = singular.train(epochs as u32);

    let model_file = workspace_root
        .join("examples")
        .join("case_a")
        .to_str()
        .unwrap()
        .to_owned()
        + "/singular_model.pt";

    singular.save_model(&model_file);

    Ok(())
}
