use candle_core::{Device, Result, Tensor};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tracing::info;

#[derive(Clone, Debug)]
pub struct NameItem {
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct NameBatch {
    pub chars: Tensor,   // Input characters [batch_size, sequence_length]
    pub targets: Tensor, // Target characters [batch_size, sequence_length]
}

#[derive(Clone)]
pub struct NameBatcher {
    device: Device,
}

impl NameBatcher {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn batch(&self, items: Vec<NameItem>) -> Result<NameBatch> {
        let max_len = items.iter().map(|item| item.name.len()).max().unwrap_or(0);
        info!("Max length: {}", max_len);

        info!("Starting to process chars...");
        let mut char_indices = Vec::with_capacity(items.len() * max_len);
        for (idx, item) in items.iter().enumerate() {
            if idx % 1000 == 0 {
                info!("Processing char tensor {} of {}", idx, items.len());
            }

            let mut sequence = vec![0i64; max_len];
            for (i, c) in item.name.chars().enumerate() {
                sequence[i] = c as u8 as i64;
            }
            char_indices.extend(sequence);
        }
        let chars = Tensor::from_vec(char_indices, (items.len(), max_len), &self.device)?;
        info!("Finished processing chars");

        info!("Starting to process targets...");
        let mut target_indices = Vec::with_capacity(items.len() * max_len);
        for (idx, item) in items.iter().enumerate() {
            if idx % 1000 == 0 {
                info!("Processing target tensor {} of {}", idx, items.len());
            }

            let mut sequence = vec![0i64; max_len];
            for (i, c) in item.name.chars().skip(1).enumerate() {
                sequence[i] = c as u8 as i64;
            }
            if item.name.len() < max_len {
                sequence[item.name.len() - 1] = 0;
            }
            target_indices.extend(sequence);
        }
        let targets = Tensor::from_vec(target_indices, (items.len(), max_len), &self.device)?;
        info!("Finished processing targets");

        Ok(NameBatch { chars, targets })
    }
}

pub fn load_names(path: &str) -> Vec<NameItem> {
    let file = File::open(path).expect("Failed to open names file");
    let reader = BufReader::new(file);

    let names: Vec<NameItem> = reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
            Some(NameItem {
                name: line.trim().to_string(),
            })
        })
        .collect();

    if names.is_empty() {
        println!("Warning: No names were loaded!");
    }

    names
}
