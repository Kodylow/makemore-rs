use crate::data::NameItem;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct BigramModel {
    pub chars: Vec<String>,
    pub char_to_idx: HashMap<String, usize>,
    pub counts: HashMap<(String, String), i32>,
    tensor: Option<Tensor>,
    probabilities: Option<Tensor>,
}

impl BigramModel {
    pub fn new(names: &[NameItem]) -> Self {
        let chars = Self::build_vocabulary(names);
        let char_to_idx = Self::create_char_mapping(&chars);

        Self {
            chars,
            char_to_idx,
            counts: HashMap::new(),
            tensor: None,
            probabilities: None,
        }
    }

    pub fn train_hashmap(&mut self, names: &[NameItem]) {
        self.counts = names
            .iter()
            .flat_map(|name| {
                let chars: Vec<_> = name.name.chars().map(|c| c.to_string()).collect();
                let tokens = std::iter::once(".".to_string())
                    .chain(chars)
                    .chain(std::iter::once(".".to_string()));
                tokens.clone().zip(tokens.skip(1))
            })
            .fold(HashMap::new(), |mut acc, (t1, t2)| {
                *acc.entry((t1, t2)).or_insert(0) += 1;
                acc
            });
    }

    pub fn train_tensor(&mut self, names: &[NameItem], device: &Device) -> Result<()> {
        let vocab_size = self.chars.len();
        let mut bigram_tensor = Tensor::zeros((vocab_size, vocab_size), DType::F32, device)?;

        for name in names {
            let tokens: Vec<String> = std::iter::once(".".to_string())
                .chain(name.name.chars().map(|c| c.to_string()))
                .chain(std::iter::once(".".to_string()))
                .collect();

            for window in tokens.windows(2) {
                let i = self.char_to_idx[&window[0]];
                let j = self.char_to_idx[&window[1]];
                let current = bigram_tensor.i((i, j))?.to_scalar::<f32>()?;
                let new_value = Tensor::new(&[[current + 1.0]], device)?;
                bigram_tensor = bigram_tensor.slice_assign(&[i..=i, j..=j], &new_value)?;
            }
        }

        self.tensor = Some(bigram_tensor);
        self.update_counts_from_tensor()?;
        Ok(())
    }

    pub fn get_tensor(&self) -> Option<&Tensor> {
        self.tensor.as_ref()
    }

    pub fn get_probabilities(&self) -> Option<&Tensor> {
        self.probabilities.as_ref()
    }

    fn update_counts_from_tensor(&mut self) -> Result<()> {
        let vocab_size = self.chars.len();
        let tensor = self.tensor.as_ref().unwrap();

        self.counts = (0..vocab_size)
            .flat_map(|i| {
                let chars = &self.chars;
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

    fn build_vocabulary(names: &[NameItem]) -> Vec<String> {
        let mut chars: Vec<String> = names
            .iter()
            .flat_map(|name| name.name.chars())
            .map(|c| c.to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .chain(std::iter::once(".".to_string()))
            .collect();

        chars.sort_by(|a, b| match (a.as_str(), b.as_str()) {
            (".", _) => std::cmp::Ordering::Less,
            (_, ".") => std::cmp::Ordering::Greater,
            _ => a.cmp(b),
        });

        chars
    }

    fn create_char_mapping(chars: &[String]) -> HashMap<String, usize> {
        chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect()
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
        let vocab_size = self.chars.len();

        let mut map = HashMap::new();
        for i in 0..vocab_size {
            for j in 0..vocab_size {
                let prob = probs.i((i, j))?.to_scalar::<f32>()?;
                if prob > 0.0 {
                    map.insert((self.chars[i].clone(), self.chars[j].clone()), prob);
                }
            }
        }
        Ok(map)
    }
}
