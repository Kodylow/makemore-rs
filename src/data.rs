//! Data structures and utilities for processing name data for neural network training.
//!
//! This module provides functionality to load, batch and prepare name data for use in
//! character-level language models and other neural network architectures that work
//! with sequences of characters.

use candle_core::{Device, Result, Tensor};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tracing::{debug, info};

/// Represents a single name item in the dataset.
///
/// This struct is used as the basic unit of data, containing a single name that can
/// be processed for neural network training.
#[derive(Clone, Debug)]
pub struct NameItem {
    /// The actual name string
    pub name: String,
}

/// A batch of processed names ready for neural network training.
///
/// Contains tensors for both input characters and their corresponding target characters,
/// structured for sequence prediction tasks where each character predicts the next.
#[derive(Clone, Debug)]
pub struct NameBatch {
    /// Input character tensors with shape [batch_size, sequence_length]
    /// Each element is the numeric representation of a character
    pub chars: Tensor,

    /// Target character tensors with shape [batch_size, sequence_length]
    /// Contains the "next character" for each position in the input,
    /// shifted by one position (target[i] = input[i+1])
    pub targets: Tensor,
}

/// Handles the conversion of name data into batched tensors for neural network training.
///
/// This struct manages the process of converting raw name strings into properly formatted
/// tensor batches, handling padding and device placement.
#[derive(Clone)]
pub struct NameBatcher {
    /// The device (CPU/GPU) where the tensors will be allocated
    device: Device,
}

impl NameBatcher {
    /// Creates a new NameBatcher that will place tensors on the specified device.
    ///
    /// # Arguments
    /// * `device` - The device (CPU/GPU) where the tensors should be allocated
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Converts a vector of NameItems into a batched tensor format suitable for training.
    ///
    /// This method:
    /// 1. Finds the longest name in the batch to determine padding length
    /// 2. Converts characters to numeric values
    /// 3. Creates input tensors where each element predicts the next character
    /// 4. Creates target tensors shifted by one position
    /// 5. Handles padding for names of different lengths
    ///
    /// # Arguments
    /// * `items` - Vector of NameItems to batch
    ///
    /// # Returns
    /// * `Result<NameBatch>` - The processed batch with input and target tensors
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
                    debug!("Processing item {} of {}", idx, items.len());
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

/// Loads names from a text file into a vector of NameItems.
///
/// Each line in the file is expected to contain a single name.
/// Empty lines and whitespace are trimmed.
///
/// # Arguments
/// * `path` - Path to the text file containing names
///
/// # Returns
/// * `Vec<NameItem>` - Vector of processed name items
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
