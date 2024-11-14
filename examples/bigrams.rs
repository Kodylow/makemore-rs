use anyhow::Result;
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use makemore_rs::plot::plot_bigram_heatmap;
use tracing::info;

fn main() -> Result<()> {
    makemore_rs::init_logging();
    let names = load_names_unique("./names.txt");
    let mut model = BigramModel::new(&names);

    model.compute_hashmap_frequencies(&names);

    info!("Bigram counts: {:?}", model.get_counts());
    plot_bigram_heatmap(
        model.get_counts(),
        model.get_chars(),
        model.get_vocabulary().get_char_to_idx(),
        "bigrams.png",
        "Bigram Counts",
    )?;
    Ok(())
}
