use anyhow::Result;
use candle_core::{DType, Device};
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use makemore_rs::plot::plot_bigram_heatmap;
use makemore_rs::utils::tensor_to_bigram_hashmap;
use tracing::info;

fn main() -> Result<()> {
    makemore_rs::utils::init_logging();
    let device = Device::Cpu;
    let names = load_names_unique("./names.txt");
    let model = BigramModel::new(&names, &device)?;
    let tensor = model.get_tensor().to_dtype(DType::F64)?;

    info!("Bigram counts: {:?}", tensor);
    plot_bigram_heatmap(
        &tensor_to_bigram_hashmap(&tensor, model.get_chars())?,
        model.get_chars(),
        model.get_vocabulary().get_char_to_idx(),
        "bigrams_tensor.png",
        "Bigram Counts",
    )?;
    Ok(())
}
