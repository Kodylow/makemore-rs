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
    count_tensor: Option<Tensor>,
    probabilities: Option<Tensor>,
}

impl BigramModel {
    /// Creates a new BigramModel with an initialized vocabulary from the given names.
    ///
    /// # Arguments
    /// * `names` - Slice of name items used to build the vocabulary
    pub fn new(names: &[NameItem]) -> Self {
        Self {
            vocabulary: Vocabulary::new(names),
            counts: HashMap::new(),
            count_tensor: None,
            probabilities: None,
        }
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

    pub fn get_tensor(&self) -> Option<&Tensor> {
        self.count_tensor.as_ref()
    }

    pub fn get_probabilities(&self) -> Option<&Tensor> {
        self.probabilities.as_ref()
    }

    pub fn get_probabilities_map(&self) -> Option<HashMap<(String, String), f32>> {
        let chars = self.vocabulary.get_chars();
        self.probabilities.as_ref().and_then(|p| {
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

    /// Computes bigram frequencies using a HashMap-based approach.
    ///
    /// # Arguments
    /// * `names` - Slice of name items to analyze
    pub fn compute_hashmap_frequencies(&mut self, names: &[NameItem]) {
        self.counts = Self::compute_bigram_counts(names);
    }

    /// Computes bigram frequencies using a tensor-based approach.
    ///
    /// # Arguments
    /// * `names` - Slice of name items to analyze
    /// * `device` - Device to store the tensor on (CPU/GPU)
    pub fn compute_tensor_frequencies(
        &mut self,
        names: &[NameItem],
        device: &Device,
    ) -> Result<()> {
        let vocab_size = self.vocabulary.get_size();
        let mut bigram_tensor = Tensor::zeros((vocab_size, vocab_size), DType::F32, device)?;

        for name in names {
            let tokens =
                Self::tokenize(&name.name.chars().map(|c| c.to_string()).collect::<Vec<_>>());

            for window in tokens.windows(2) {
                let char_to_idx = self.vocabulary.get_char_to_idx();
                let i = char_to_idx[&window[0]];
                let j = char_to_idx[&window[1]];
                let current = bigram_tensor.i((i, j))?.to_scalar::<f32>()?;
                let new_value = Tensor::new(&[[current + 1.0]], device)?;
                bigram_tensor = bigram_tensor.slice_assign(&[i..=i, j..=j], &new_value)?;
            }
        }

        self.count_tensor = Some(bigram_tensor);
        self.sync_counts_from_tensor()?;
        Ok(())
    }

    /// Converts raw frequencies into probabilities by normalizing each row
    /// to sum to 1.0.
    pub fn compute_probabilities(&mut self) -> Result<()> {
        let tensor = self.count_tensor.as_ref().unwrap();
        let probs = tensor.to_dtype(DType::F32)?;
        let row_sums = probs.sum_keepdim(1)?;
        let normalized = probs.broadcast_div(&row_sums)?;
        self.probabilities = Some(normalized);
        Ok(())
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

    fn compute_bigram_counts(names: &[NameItem]) -> HashMap<(String, String), i32> {
        names
            .iter()
            .flat_map(|name| {
                let chars: Vec<_> = name.name.chars().map(|c| c.to_string()).collect();
                let tokens = Self::tokenize(&chars).into_iter();
                tokens.clone().zip(tokens.skip(1))
            })
            .fold(HashMap::new(), |mut acc, (t1, t2)| {
                *acc.entry((t1, t2)).or_insert(0) += 1;
                acc
            })
    }

    fn tokenize(chars: &[String]) -> Vec<String> {
        std::iter::once(".".to_string())
            .chain(chars.iter().cloned())
            .chain(std::iter::once(".".to_string()))
            .collect()
    }

    fn sync_counts_from_tensor(&mut self) -> Result<()> {
        let vocab_size = self.vocabulary.get_size();
        let tensor = self.count_tensor.as_ref().unwrap();

        self.counts = (0..vocab_size)
            .flat_map(|i| {
                let chars = self.vocabulary.get_chars();
                (0..vocab_size).filter_map(move |j| {
                    let count = tensor.i((i, j)).as_ref().ok()?.to_scalar::<f32>().ok()? as i32;
                    if count > 0 {
                        Some(((chars[i].clone(), chars[j].clone()), count))
                    } else {
                        None
                    }
                })
            })
            .collect();

        Ok(())
    }
}
