use candle_core::Device;
use makemore_rs::data::{load_names, NameBatcher};
use tracing::{info, warn};

/// The goal is to maximize the likelihood of the data with respect to model parameters
/// (statistical modeling). This is equivalent to maximizing the log likelihood (because log
/// is monotonic), which is equivalent to minimizing the negative log likelihood, which is
/// equivalent to minimizing the average negative log likelihood.

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    Ok(())
}
