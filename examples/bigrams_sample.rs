use anyhow::Result;
use candle_core::{Device, Tensor};
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use tracing::info;

fn main() -> Result<()> {
    makemore_rs::init_logging();
    let device = Device::Cpu;

    // Create a simple probability distribution like in the PyTorch example
    let probs = Tensor::new(&[0.6064_f32, 0.3033, 0.0903], &device)?;
    info!("Probs: {:?}", probs);
    // Create and train the model
    let names = load_names_unique("./names.txt");
    let mut model = BigramModel::new(&names);
    model.train_tensor(&names, &device)?;
    model.normalize_probabilities()?;

    // Test basic multinomial sampling
    info!("Sampling from test distribution:");
    let samples = model.multinomial(&probs, 100, true)?;
    info!("Sample results: {:?}", samples.to_vec1::<i64>()?);

    Ok(())
}
