use std::collections::HashSet;
use std::{collections::HashMap, fs};

use burn::autodiff::ADBackendDecorator;
use burn::backend::{wgpu::AutoGraphicsApi, WgpuBackend};
use burn::tensor::{Data, Int, Tensor};

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
}

fn build_bigrams(names: Vec<&str>) -> Vec<String> {
    let mut bigrams: Vec<String> = Vec::new();
    for name in &names {
        // hallucinate a start character '<S>' and end character '<E>'
        let name = format!("{}{}{}", SPECIAL, name, SPECIAL);
        let mut chars = name.chars();
        let mut prev = chars.next().unwrap();

        for c in chars {
            bigrams.push(format!("{}{}", prev, c));
            prev = c;
        }
    }
    bigrams
}

fn build_tensors(
    bigram_counts: HashMap<String, i32>,
    stoi: HashMap<char, usize>,
) -> Tensor<MyAutodiffBackend, 2, Int> {
    let mut data: Vec<Vec<i32>> = vec![vec![0; UNIQUE_CHAR_COUNT]; UNIQUE_CHAR_COUNT];

    let data: [[i32; UNIQUE_CHAR_COUNT]; UNIQUE_CHAR_COUNT] = data
        .into_iter()
        .map(|v| {
            let mut arr = [0; UNIQUE_CHAR_COUNT];
            for (i, val) in v.into_iter().enumerate() {
                arr[i] = val;
            }
            arr
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let data = Data::from(data);
    let tensor = Tensor::from_data(data);

    tensor
}

fn build_chars(names: &Vec<&str>) -> Vec<char> {
    let mut set_of_chars: HashSet<char> = names.iter().flat_map(|name| name.chars()).collect();

    // add start and end characters
    set_of_chars.insert(SPECIAL);

    // convert to vector and sort
    let mut set_of_chars: Vec<char> = set_of_chars.into_iter().collect();
    set_of_chars.sort();

    set_of_chars
}
