use std::{collections::HashMap, fs};

const START: char = '<';
const END: char = '>';

fn main() {
    let contents =
        fs::read_to_string("./makemore/names.txt").expect("Something went wrong reading the file");
    let names: Vec<&str> = contents.lines().collect();

    // create character bigrams
    let mut bigrams: Vec<String> = Vec::new();
    for name in &names {
        // hallucinate a start character '<S>' and end character '<E>'
        let name = format!("{}{}{}", START, name, END);
        let mut chars = name.chars();
        let mut prev = chars.next().unwrap();

        // create bigrams
        for c in chars {
            bigrams.push(format!("{}{}", prev, c));
            prev = c;
        }
    }

    let mut bigram_map: HashMap<String, i32> = HashMap::new();
    for bigram in &bigrams {
        *bigram_map.entry(bigram.to_string()).or_default() += 1;
    }
    let mut bigram_counts: Vec<(String, i32)> = bigram_map.into_iter().collect();
    bigram_counts.sort_by_key(|&(_, count)| -count);
    for (bigram, count) in bigram_counts {
        println!("{}: {}", bigram, count);
    }
}
