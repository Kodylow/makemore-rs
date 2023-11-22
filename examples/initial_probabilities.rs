use std::{collections::HashMap, fs};

use makemore_rs::{build_bigrams, build_chars, build_tensors};

const FILE_PATH: &str = "./makemore/names.txt";

// Gets through timestamp 0:50 on makemore part 1
fn main() {
    let contents = fs::read_to_string(FILE_PATH).expect("Something went wrong reading the file");
    let names: Vec<&str> = contents.lines().collect();

    // create a sorted set of unique characters
    let sorted_chars = build_chars(&names);
    let stoi: HashMap<char, usize> = sorted_chars
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();
    let itos: HashMap<usize, char> = stoi.iter().map(|(c, i)| (*i, *c)).collect();

    // create character bigrams
    let bigrams = build_bigrams(names);

    // create hashmap of bigram counts
    let bigram_counts: HashMap<String, f32> =
        bigrams.iter().fold(HashMap::new(), |mut acc, bigram| {
            *acc.entry(bigram.to_string()).or_default() += 1.0;
            acc
        });
    println!("bigram_counts: {:?}", bigram_counts);

    // create a normalized tensor of bigram probabilities
    // burn automatically broadcasts to the correct shape which is sweet
    let bigrams_tensor = build_tensors(bigram_counts, stoi);
    let p = bigrams_tensor.slice([0..1]);
    let p = p.clone().div_scalar(p.sum().into_scalar());

    // Gets through timestamp 0:50 on the 2nd karpathy video
    println!("Probabilities that letter starts a word: {:?}", p.to_data());
}
