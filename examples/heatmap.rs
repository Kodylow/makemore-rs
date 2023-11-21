use std::collections::HashSet;
use std::{collections::HashMap, fs};

use burn::autodiff::ADBackendDecorator;
use burn::backend::{wgpu::AutoGraphicsApi, WgpuBackend};
use burn::tensor::{Int, Tensor};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::Rectangle;
use plotters::style::Color;
use plotters::style::IntoFont;
use plotters::style::{RGBColor, WHITE};

const START: char = '<';
const END: char = '>';

const UNIQUE_CHAR_COUNT: usize = 28;

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;
    let n: Tensor<MyAutodiffBackend, 2, Int> =
        Tensor::zeros([UNIQUE_CHAR_COUNT, UNIQUE_CHAR_COUNT]);

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
        let name = format!("{}{}{}", START, name, END);
        let mut chars = name.chars();
        let mut prev = chars.next().unwrap();

        // create bigrams
        for c in chars {
            bigrams.push(format!("{}{}", prev, c));
            prev = c;
        }
    }

    // print a heatmap of bigram counts using plotters
    // Count bigrams
    let mut bigram_counts: HashMap<String, usize> = HashMap::new();
    for bigram in bigrams {
        *bigram_counts.entry(bigram).or_insert(0) += 1;
    }

    // Normalize counts to range [0, 1]
    let max_count = *bigram_counts.values().max().unwrap() as f32;
    let normalized_counts: HashMap<String, f32> = bigram_counts
        .iter()
        .map(|(bigram, count)| (bigram.clone(), *count as f32 / max_count))
        .collect();

    // Create heatmap
    let root = BitMapBackend::new("heatmap.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Heatmap of Bigram Counts", ("Arial", 50).into_font())
        .margin(5)
        .set_all_label_area_size(50)
        .build_cartesian_2d(0..UNIQUE_CHAR_COUNT, 0..UNIQUE_CHAR_COUNT)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(UNIQUE_CHAR_COUNT)
        .y_labels(UNIQUE_CHAR_COUNT)
        .draw()
        .unwrap();

    for (bigram, count) in normalized_counts {
        let x = stoi[&bigram.chars().nth(0).unwrap()];
        let y = stoi[&bigram.chars().nth(1).unwrap()];
        let color = ((1.0 - count) * 255.0) as u8;
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x, y), (x + 1, y + 1)],
                RGBColor(color, color, color).filled(),
            )))
            .unwrap();
    }

    // Save heatmap
    root.present().unwrap();
}

fn get_chars(names: &Vec<&str>) -> Vec<char> {
    let mut set_of_chars: HashSet<char> = names.iter().flat_map(|name| name.chars()).collect();

    // add start and end characters
    set_of_chars.insert(START);
    set_of_chars.insert(END);

    // convert to vector and sort
    let mut set_of_chars: Vec<char> = set_of_chars.into_iter().collect();
    set_of_chars.sort();

    set_of_chars
}
