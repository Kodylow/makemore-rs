use anyhow::Result;
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use makemore_rs::plot::plot_bigram_heatmap;
use tracing::info;

fn main() -> Result<()> {
    let names = load_names_unique("./names.txt");
    let mut model = BigramModel::new(&names);

    model.train_hashmap(&names);

    info!("Bigram counts: {:?}", model.counts);
    plot_bigram_heatmap(
        &model.counts,
        &model.chars,
        &model.char_to_idx,
        "bigrams.png",
        "Bigram Counts",
    )?;
    Ok(())
}
