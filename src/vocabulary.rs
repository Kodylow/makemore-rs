use std::collections::{HashMap, HashSet};

use crate::data::NameItem;

/// A vocabulary that maps characters to indices and vice versa.
///
/// This struct maintains a mapping between characters and their corresponding indices,
/// which is useful for converting text data into numeric form for machine learning models.
/// It automatically builds a vocabulary from a list of names, ensuring that all unique
/// characters are accounted for.
///
/// The vocabulary always includes a special "." character that is guaranteed to be at index 0,
/// which can be used as a start/end token or padding character.
///
/// # Examples
///
/// ```
/// use makemore_rs::data::NameItem;
/// use makemore_rs::vocabulary::Vocabulary;
///
/// let names = vec![
///     NameItem { name: "Alice".to_string() },
///     NameItem { name: "Bob".to_string() }
/// ];
///
/// let vocab = Vocabulary::new(&names);
/// assert!(vocab.get_chars().contains(&".".to_string()));
/// assert!(vocab.get_chars().contains(&"A".to_string()));
/// ```
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Vector of unique characters in the vocabulary, sorted alphabetically with "." first
    chars: Vec<String>,
    /// Mapping from characters to their corresponding indices in the vocabulary
    char_to_idx: HashMap<String, usize>,
}

impl Vocabulary {
    /// Creates a new vocabulary from a slice of name items.
    ///
    /// This method extracts all unique characters from the provided names,
    /// sorts them alphabetically (with "." always first), and builds the
    /// character-to-index mapping.
    ///
    /// # Arguments
    ///
    /// * `names` - A slice of NameItems to build the vocabulary from
    ///
    /// # Returns
    ///
    /// A new Vocabulary instance containing all unique characters from the names
    pub fn new(names: &[NameItem]) -> Self {
        let chars = Self::build_chars(names);
        let char_to_idx = chars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        Self { chars, char_to_idx }
    }

    /// Builds a sorted vector of unique characters from the provided names.
    ///
    /// This method:
    /// 1. Extracts all unique characters from the names
    /// 2. Adds a "." character (useful for start/end tokens or padding)
    /// 3. Sorts the characters alphabetically, with "." always first
    ///
    /// # Arguments
    ///
    /// * `names` - A slice of NameItems to extract characters from
    ///
    /// # Returns
    ///
    /// A sorted vector of unique characters as Strings
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

    /// Returns a reference to the vector of characters in the vocabulary.
    ///
    /// The characters are sorted alphabetically with "." always first.
    pub fn get_chars(&self) -> &Vec<String> {
        &self.chars
    }

    /// Returns the size of the vocabulary (number of unique characters).
    pub fn get_size(&self) -> usize {
        self.chars.len()
    }

    /// Returns a reference to the character-to-index mapping.
    ///
    /// This mapping can be used to convert characters to their corresponding
    /// indices in the vocabulary.
    pub fn get_char_to_idx(&self) -> &HashMap<String, usize> {
        &self.char_to_idx
    }
}
