use anyhow::Result;
use candle_core::Device;
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use makemore_rs::plot::plot_bigram_heatmap;
use tracing::info;

fn main() -> Result<()> {
    makemore_rs::init_logging();
    let device = Device::Cpu;
    let names = load_names_unique("./names.txt");
    let mut model = BigramModel::new(&names);

    model.compute_tensor_frequencies(&names, &device)?;

    info!("Bigram counts: {:?}", model.get_counts());
    plot_bigram_heatmap(
        model.get_counts(),
        model.get_chars(),
        model.get_vocabulary().get_char_to_idx(),
        "bigrams_tensor.png",
        "Bigram Counts",
    )?;
    Ok(())
}
