use candle_core::Device;
use data::{load_names, NameBatcher};
use tracing::{info, warn};

mod data;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Loading names from file");
    let names = load_names("./names.txt");
    info!("Loaded {} names", names.len());

    if names.is_empty() {
        warn!("No names were loaded - cannot proceed with empty dataset");
        return Ok(());
    }

    info!("Creating device");
    let device = Device::cuda_if_available(0)?;

    info!("Creating batcher");
    let batcher = NameBatcher::new(device);

    info!("Creating batch from {} names", names.len());
    let batch = batcher.batch(names)?;
    info!("Batch created successfully");
    println!("{:?}", batch);

    Ok(())
}
