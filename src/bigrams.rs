use crate::data::NameItem;
use crate::vocabulary::Vocabulary;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use rand::Rng;
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone)]
pub struct BigramModel {
    vocabulary: Vocabulary,
    counts: HashMap<(String, String), i32>,
    tensor: Option<Tensor>,
    probabilities: Option<Tensor>,
}

impl BigramModel {
    pub fn new(names: &[NameItem]) -> Self {
        Self {
            vocabulary: Vocabulary::new(names),
            counts: HashMap::new(),
            tensor: None,
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
        self.tensor.as_ref()
    }

    pub fn get_probabilities(&self) -> Option<&Tensor> {
        self.probabilities.as_ref()
    }

    pub fn train_hashmap(&mut self, names: &[NameItem]) {
        self.counts = Self::compute_bigram_counts(names);
    }

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

    pub fn train_tensor(&mut self, names: &[NameItem], device: &Device) -> Result<()> {
        let vocab_size = self.vocabulary.get_size();
        let mut bigram_tensor = Tensor::zeros((vocab_size, vocab_size), DType::F32, device)?;

        for name in names {
            let tokens: Vec<String> = std::iter::once(".".to_string())
                .chain(name.name.chars().map(|c| c.to_string()))
                .chain(std::iter::once(".".to_string()))
                .collect();

            for window in tokens.windows(2) {
                let char_to_idx = self.vocabulary.get_char_to_idx();
                let i = char_to_idx[&window[0]];
                let j = char_to_idx[&window[1]];
                let current = bigram_tensor.i((i, j))?.to_scalar::<f32>()?;
                let new_value = Tensor::new(&[[current + 1.0]], device)?;
                bigram_tensor = bigram_tensor.slice_assign(&[i..=i, j..=j], &new_value)?;
            }
        }

        self.tensor = Some(bigram_tensor);
        self.update_counts_from_tensor()?;
        Ok(())
    }

    fn update_counts_from_tensor(&mut self) -> Result<()> {
        let vocab_size = self.vocabulary.get_size();
        let tensor = self.tensor.as_ref().unwrap();

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

    pub fn normalize_probabilities(&mut self) -> Result<()> {
        let tensor = self.tensor.as_ref().unwrap();

        // Convert to f32
        let probs = tensor.to_dtype(DType::F32)?;

        // Sum along rows and broadcast for division
        let row_sums = probs.sum_keepdim(1)?;
        let normalized = probs.broadcast_div(&row_sums)?;

        // Store the normalized probabilities
        self.probabilities = Some(normalized);
        Ok(())
    }

    pub fn get_probabilities_map(&self) -> Result<HashMap<(String, String), f32>> {
        let probs = self.probabilities.as_ref().unwrap();
        let vocab_size = self.vocabulary.get_size();

        let mut map = HashMap::new();
        for i in 0..vocab_size {
            for j in 0..vocab_size {
                let prob = probs.i((i, j))?.to_scalar::<f32>()?;
                if prob > 0.0 {
                    map.insert(
                        (
                            self.vocabulary.get_chars()[i].clone(),
                            self.vocabulary.get_chars()[j].clone(),
                        ),
                        prob,
                    );
                }
            }
        }
        Ok(map)
    }

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
}
