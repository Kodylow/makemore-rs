use std::collections::HashSet;
use std::{collections::HashMap, fs};

use burn::autodiff::ADBackendDecorator;
use burn::backend::{wgpu::AutoGraphicsApi, WgpuBackend};
use burn::tensor::{Int, Tensor};

const SPECIAL: char = '.';

const UNIQUE_CHAR_COUNT: usize = 28;

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let contents =
        fs::read_to_string("./makemore/names.txt").expect("Something went wrong reading the file");
    let names: Vec<&str> = contents.lines().collect();

    // create a sorted set of unique characters
    let sorted_chars = get_chars(&names);
    let stoi: HashMap<char, usize> = sorted_chars
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();
    let itos: HashMap<usize, char> = sorted_chars
        .iter()
        .enumerate()
        .map(|(i, c)| (i, *c))
        .collect();

    // create character bigrams
    let mut bigrams: Vec<String> = Vec::new();
    for name in &names {
        // hallucinate a start character '<S>' and end character '<E>'
        let name = format!("{}{}{}", SPECIAL, name, SPECIAL);
        let mut chars = name.chars();
        let mut prev = chars.next().unwrap();

        // create bigrams
        for c in chars {
            bigrams.push(format!("{}{}", prev, c));
            prev = c;
        }
    }

    // create 2 dimensional tensor of bigram counts
    let mut bigram_counts: Tensor<MyAutodiffBackend, 2, Int> =
        Tensor::zeros([UNIQUE_CHAR_COUNT, UNIQUE_CHAR_COUNT]);
    for bigram in &bigrams {
        let i = stoi[&bigram.chars().nth(0).unwrap()];
        let j = stoi[&bigram.chars().nth(1).unwrap()];
        let old_value = bigram_counts.clone().slice([i..i + 1, j..j + 1]);
        let new_value = old_value + 1;
        bigram_counts = bigram_counts.slice_assign([i..i + 1, j..j + 1], new_value);
    }
}

fn get_chars(names: &Vec<&str>) -> Vec<char> {
    let mut set_of_chars: HashSet<char> = names.iter().flat_map(|name| name.chars()).collect();

    // add start and end characters
    set_of_chars.insert(SPECIAL);

    // convert to vector and sort
    let mut set_of_chars: Vec<char> = set_of_chars.into_iter().collect();
    set_of_chars.sort();

    set_of_chars
}
