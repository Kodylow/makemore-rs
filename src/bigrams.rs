//! Bigram language model implementation that tracks character pair frequencies
//! and their probabilities in a given dataset.

use crate::data::NameItem;
use crate::vocabulary::Vocabulary;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use rand::Rng;
use std::collections::HashMap;
use tracing::info;

/// A statistical model that captures the frequencies and probabilities
/// of character pairs (bigrams) in text data.
#[derive(Debug, Clone)]
pub struct BigramModel {
    vocabulary: Vocabulary,
    counts: HashMap<(String, String), i32>,
    count_tensor: Tensor,
    probabilities: Tensor,
}

impl BigramModel {
    /// Creates a new BigramModel with computed frequencies and probabilities
    ///
    /// # Arguments
    /// * `names` - Slice of name items used to build the vocabulary
    /// * `device` - Device to store tensors on (CPU/GPU)
    pub fn new(names: &[NameItem], device: &Device) -> Result<Self> {
        let vocabulary = Vocabulary::new(names);
        let vocab_size = vocabulary.get_size();

        // Initialize and compute count tensor
        let mut count_tensor = Tensor::zeros((vocab_size, vocab_size), DType::F32, device)?;

        for name in names {
            let tokens =
                Self::tokenize(&name.name.chars().map(|c| c.to_string()).collect::<Vec<_>>());
            for window in tokens.windows(2) {
                let char_to_idx = vocabulary.get_char_to_idx();
                let i = char_to_idx[&window[0]];
                let j = char_to_idx[&window[1]];
                let current = count_tensor.i((i, j))?.to_scalar::<f32>()?;
                let new_value = Tensor::new(&[[current + 1.0]], device)?;
                count_tensor = count_tensor.slice_assign(&[i..=i, j..=j], &new_value)?;
            }
        }

        // Compute probabilities
        let probs = count_tensor.to_dtype(DType::F32)?;
        let row_sums = probs.sum_keepdim(1)?;
        let probabilities = probs.broadcast_div(&row_sums)?;

        // Compute hashmap counts
        let counts = (0..vocab_size)
            .flat_map(|i| {
                let count_tensor = &count_tensor;
                let chars = vocabulary.get_chars();
                (0..vocab_size).filter_map(move |j| {
                    let count = count_tensor
                        .i((i, j))
                        .as_ref()
                        .ok()?
                        .to_scalar::<f32>()
                        .ok()? as i32;
                    if count > 0 {
                        Some(((chars[i].clone(), chars[j].clone()), count))
                    } else {
                        None
                    }
                })
            })
            .collect();

        Ok(Self {
            vocabulary,
            counts,
            count_tensor,
            probabilities,
        })
    }

    pub fn get_vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    pub fn get_counts(&self) -> &HashMap<(String, String), i32> {
        &self.counts
    }

    pub fn get_chars(&self) -> &Vec<String> {
        self.vocabulary.get_chars()
    }

    pub fn get_tensor(&self) -> &Tensor {
        &self.count_tensor
    }

    pub fn get_probabilities(&self) -> &Tensor {
        &self.probabilities
    }

    pub fn get_probabilities_map(&self) -> Option<HashMap<(String, String), f32>> {
        let probabilities = &self.probabilities;
        let chars = self.vocabulary.get_chars();
        probabilities.to_dtype(DType::F32).ok().and_then(|p| {
            p.to_dtype(DType::F32)
                .ok()?
                .to_vec2::<f32>()
                .ok()
                .map(|data| {
                    data.iter()
                        .enumerate()
                        .flat_map(|(i, row)| {
                            row.iter()
                                .enumerate()
                                .map(move |(j, &v)| ((chars[i].clone(), chars[j].clone()), v))
                        })
                        .collect()
                })
        })
    }

    /// Samples indices from a probability distribution using the multinomial distribution.
    ///
    /// # Arguments
    /// * `probs` - Tensor containing probabilities
    /// * `num_samples` - Number of samples to draw
    /// * `replacement` - Whether to sample with replacement
    ///
    /// # Returns
    /// * Tensor containing sampled indices
    pub fn multinomial(
        &self,
        probs: &Tensor,
        num_samples: i64,
        replacement: bool,
    ) -> Result<Tensor> {
        info!(
            "Starting multinomial sampling with {} samples, replacement: {}",
            num_samples, replacement
        );
        let device = probs.device();
        let mut p = if probs.dims().len() > 1 {
            probs.flatten_all()?.to_vec1::<f32>()?
        } else {
            probs.to_vec1::<f32>()?
        };

        if !replacement && num_samples > p.len() as i64 {
            return Err(anyhow::anyhow!(
                "Cannot sample {} items without replacement from tensor of length {}",
                num_samples,
                p.len()
            ));
        }

        let mut samples = Vec::with_capacity(num_samples as usize);
        let mut rng = rand::thread_rng();

        // Create cumulative probabilities once
        let mut cumulative = vec![0.0; p.len()];
        let mut sum = 0.0;
        for (i, &prob) in p.iter().enumerate() {
            sum += prob;
            cumulative[i] = sum;
        }

        for _ in 0..num_samples {
            let r: f32 = rng.gen::<f32>() * sum;

            // Binary search for the index
            let selected_idx =
                match cumulative.binary_search_by(|&cum| cum.partial_cmp(&r).unwrap()) {
                    Ok(idx) => idx,
                    Err(idx) => idx,
                };

            samples.push(selected_idx as i64);

            if !replacement {
                // Update cumulative probabilities
                sum -= p[selected_idx];
                p[selected_idx] = 0.0;
                let mut running_sum = 0.0;
                for (i, &prob) in p.iter().enumerate() {
                    running_sum += prob;
                    cumulative[i] = running_sum;
                }
            }
        }

        Tensor::new(samples.as_slice(), device).map_err(|e| e.into())
    }

    // Private helper methods below

    fn tokenize(chars: &[String]) -> Vec<String> {
        std::iter::once(".".to_string())
            .chain(chars.iter().cloned())
            .chain(std::iter::once(".".to_string()))
            .collect()
    }
}
