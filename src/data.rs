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

        let (chars, targets) = items.iter().enumerate().fold(
            (
                Vec::with_capacity(items.len() * max_len),
                Vec::with_capacity(items.len() * max_len),
            ),
            |(mut chars, mut targets), (idx, item)| {
                if idx % 1000 == 0 {
                    info!("Processing item {} of {}", idx, items.len());
                }

                let mut char_seq = vec![0i64; max_len];
                let mut target_seq = vec![0i64; max_len];

                for (i, c) in item.name.chars().enumerate() {
                    char_seq[i] = c as i64;
                }

                for (i, c) in item.name.chars().skip(1).enumerate() {
                    target_seq[i] = c as i64;
                }

                chars.extend(char_seq);
                targets.extend(target_seq);
                (chars, targets)
            },
        );

        let chars = Tensor::from_vec(chars, (items.len(), max_len), &self.device)?;
        let targets = Tensor::from_vec(targets, (items.len(), max_len), &self.device)?;

        Ok(NameBatch { chars, targets })
    }
}

pub fn load_names(path: &str) -> Vec<NameItem> {
    BufReader::new(File::open(path).expect("Failed to open names file"))
        .lines()
        .filter_map(|line| {
            line.ok().map(|l| NameItem {
                name: l.trim().to_string(),
            })
        })
        .collect()
}
