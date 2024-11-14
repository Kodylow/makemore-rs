use std::collections::{HashMap, HashSet};

use crate::data::NameItem;

#[derive(Debug, Clone)]
pub struct Vocabulary {
    chars: Vec<String>,
    char_to_idx: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn new(names: &[NameItem]) -> Self {
        let chars = Self::build_chars(names);
        let char_to_idx = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        Self { chars, char_to_idx }
    }

    pub fn build_chars(names: &[NameItem]) -> Vec<String> {
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

    pub fn get_chars(&self) -> &Vec<String> {
        &self.chars
    }

    pub fn get_size(&self) -> usize {
        self.chars.len()
    }

    pub fn get_char_to_idx(&self) -> &HashMap<String, usize> {
        &self.char_to_idx
    }
}
