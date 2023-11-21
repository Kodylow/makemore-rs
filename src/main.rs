use std::collections::HashSet;
use std::{collections::HashMap, fs};

use burn::autodiff::ADBackendDecorator;
use burn::backend::{wgpu::AutoGraphicsApi, WgpuBackend};
use burn::tensor::{Int, Tensor};

const SPECIAL: char = '.';
const FILE_PATH: &str = "./makemore/names.txt";
const UNIQUE_CHAR_COUNT: usize = 28;

type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

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

    // create character bigrams
    let bigrams = build_bigrams(names);

    // create hashmap of bigram counts
    let bigram_counts: HashMap<String, i32> =
        bigrams.iter().fold(HashMap::new(), |mut acc, bigram| {
            *acc.entry(bigram.to_string()).or_default() += 1;
            acc
        });

    // create a tensor for each first char like Tensor(3,5,9) for aa:3, ab:5, ac:9
    let bigram_tensor = build_tensors(bigram_counts, stoi);

    println!("{:?}", bigram_tensor.to_data());
}
