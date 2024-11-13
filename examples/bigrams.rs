use anyhow::Result;
use makemore_rs::data::load_names;
use makemore_rs::plot::plot_bigram_heatmap;
use std::collections::{HashMap, HashSet};
use tracing::info;

fn main() -> Result<()> {
    let mut b: HashMap<(String, String), i32> = HashMap::new();
    let names = load_names("./names.txt");

    for name in names {
        let char_strings: Vec<String> = name.name.chars().map(|c| c.to_string()).collect();
        let tokens = std::iter::once("<S>")
            .chain(char_strings.iter().map(|s| s.as_str()))
            .chain(std::iter::once("<E>"));

        for (t1, t2) in tokens.clone().zip(tokens.skip(1)) {
            let bigram = (t1.to_string(), t2.to_string());
            *b.entry(bigram).or_insert(0) += 1;
        }
    }

    info!("Bigram counts: {:?}", b);

    // Get unique characters for axes
    let mut chars: HashSet<String> = HashSet::new();
    for (ch1, ch2) in b.keys() {
        chars.insert(ch1.clone());
        chars.insert(ch2.clone());
    }
    let chars: Vec<String> = chars.into_iter().collect();

    // Create mapping from char to index
    let char_to_idx: HashMap<String, usize> = chars
        .iter()
        .enumerate()
        .map(|(i, c)| (c.clone(), i))
        .collect();

    plot_bigram_heatmap(&b, &chars, &char_to_idx, "bigrams.png")?;
    Ok(())
}
