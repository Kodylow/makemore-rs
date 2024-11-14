use anyhow::Result;
use candle_core::Device;
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use makemore_rs::plot::plot_bigram_heatmap;
use tracing::info;

fn main() -> Result<()> {
    makemore_rs::utils::init_logging();
    let device = Device::Cpu;
    let names = load_names_unique("./names.txt");
    let model = BigramModel::new(&names, &device)?;

    info!("Bigram probabilities: {:?}", model.get_probabilities());
    plot_bigram_heatmap(
        &model.get_probabilities_map().unwrap(),
        model.get_chars(),
        model.get_vocabulary().get_char_to_idx(),
        "bigrams_probabilities.png",
        "Bigram Probabilities",
    )?;
    Ok(())
}
