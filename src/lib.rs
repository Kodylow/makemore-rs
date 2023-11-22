use std::collections::{HashMap, HashSet};

use burn::autodiff::ADBackendDecorator;
use burn::backend::{wgpu::AutoGraphicsApi, WgpuBackend};
use burn::tensor::{Float, Tensor};

type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

const SPECIAL: char = '.';
const UNIQUE_CHAR_COUNT: usize = 27;

pub fn build_bigrams(names: Vec<&str>) -> Vec<String> {
    let mut bigrams: Vec<String> = Vec::new();
    for name in &names {
        // hallucinate a start character and end character
        let name = format!("{}{}{}", SPECIAL, name, SPECIAL);
        let mut chars = name.chars();
        let mut prev = chars.next().unwrap();

        for c in chars {
            let bigram = format!("{}{}", prev, c);
            bigrams.push(bigram);
            prev = c;
        }
    }
    bigrams
}

pub fn build_tensors(
    bigram_counts: HashMap<String, f32>,
    stoi: HashMap<char, usize>,
) -> Tensor<MyAutodiffBackend, 2, Float> {
    let mut bigram_tensor: Tensor<MyAutodiffBackend, 2, Float> =
        Tensor::zeros([UNIQUE_CHAR_COUNT, UNIQUE_CHAR_COUNT]);
    for (bigram, count) in bigram_counts {
        let chars: Vec<char> = bigram.chars().collect();
        let i = stoi[&chars[0]];
        let j = stoi[&chars[1]];
        let new_value = bigram_tensor.clone().slice([i..i + 1, j..j + 1]) + count;
        bigram_tensor = bigram_tensor.slice_assign([i..i + 1, j..j + 1], new_value);
    }

    let bigram_tensor = bigram_tensor
        .clone()
        .div_scalar(bigram_tensor.sum().into_scalar());

    bigram_tensor
}

pub fn build_chars(names: &Vec<&str>) -> Vec<char> {
    let mut set_of_chars: HashSet<char> = names.iter().flat_map(|name| name.chars()).collect();

    // add start and end characters
    set_of_chars.insert(SPECIAL);

    // convert to vector and sort
    let mut set_of_chars: Vec<char> = set_of_chars.into_iter().collect();
    set_of_chars.sort();

    set_of_chars
}
