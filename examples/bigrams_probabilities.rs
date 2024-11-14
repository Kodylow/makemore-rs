use anyhow::Result;
use candle_core::Device;
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use makemore_rs::plot::plot_bigram_heatmap;
use tracing::info;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let names = load_names_unique("./names.txt");
    let mut model = BigramModel::new(&names);

    model.train_tensor(&names, &device)?;
    model.normalize_probabilities()?;

    info!("Bigram probabilities: {:?}", model.get_probabilities_map()?);
    plot_bigram_heatmap(
        &model.get_probabilities_map()?,
        &model.chars,
        &model.char_to_idx,
        "bigrams_probabilities.png",
        "Bigram Probabilities",
    )?;
    Ok(())
}
