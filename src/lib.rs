use candle_core::{DType, Device, IndexOp, Tensor};

pub mod bigrams;
pub mod data;
pub mod plot;
pub mod utils;
pub mod vocabulary;

/// Creates bigram pairs of consecutive characters from input words, converting them to indices
///
/// Each word is padded with '.' at start and end. Characters are converted to indices where:
/// - '.' = 0
/// - 'a' to 'z' = 1 to 26
///
/// # Arguments
/// * `words` - Slice of strings to process
///
/// # Returns
/// * Tuple of (input indices, target indices) for training
pub fn create_character_pairs(
    words: &[String],
) -> Result<(Vec<i64>, Vec<i64>), Box<dyn std::error::Error>> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for word in &words[..1] {
        let chars: Vec<char> = format!(".{}.", word).chars().collect();
        for window in chars.windows(2) {
            let (ch1, ch2) = (window[0], window[1]);
            let ix1 = char_to_index(ch1);
            let ix2 = char_to_index(ch2);

            println!("{} {}", ch1, ch2);
            xs.push(ix1);
            ys.push(ix2);
        }
    }

    Ok((xs, ys))
}

/// Creates one-hot encoded vectors from input indices
///
/// One-hot encoding converts categorical data (like character indices) into a binary vector format
/// that neural networks can process. For each input index, it creates a vector of zeros with length
/// equal to vocabulary size, setting a single 1 at the index position.
///
/// This encoding is necessary because neural networks can't directly work with categorical indices.
/// The one-hot representation allows the network to:
/// - Learn separate weights for each category
/// - Make independent predictions for each possible class
/// - Avoid imposing artificial ordering between categories
///
/// # Arguments
/// * `xs` - Input tensor containing indices
/// * `num_classes` - Number of possible classes (vocabulary size)
/// * `device` - Device to store tensors on (CPU/GPU)
///
/// # Returns
/// * Tensor of one-hot encoded vectors
pub fn create_one_hot_encoding(
    xs: &Tensor,
    num_classes: usize,
    device: &Device,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let xs_zeros = Tensor::zeros((xs.dim(0)?, num_classes), DType::F32, device)?;
    let indices = xs.to_dtype(DType::I64)?.unsqueeze(1)?;
    let ones = Tensor::ones(indices.shape(), DType::F32, device)?;
    let x_one_hot = xs_zeros.scatter_add(&indices, &ones, 1)?;
    Ok(x_one_hot)
}

/// Converts a character to its corresponding index
///
/// The '.' character is used as a special token to mark the start and end of words.
/// This helps the model learn word boundaries and valid character transitions at
/// the beginning and end of names. For example, in the word "emma", we add dots
/// to get ".emma.", allowing the model to learn:
/// - What characters commonly start words (. -> e)
/// - What characters commonly end words (a -> .)
/// - That words have clear boundaries
///
/// # Arguments
/// * `c` - Character to convert
///
/// # Returns
/// * Index value as i64
///
/// # Panics
/// * If character is not '.' or lowercase a-z
pub fn char_to_index(c: char) -> i64 {
    match c {
        '.' => 0,
        'a'..='z' => (c as u8 - b'a' + 1) as i64,
        _ => panic!("Unexpected character: {}", c),
    }
}

/// Verifies that manual dot product calculation matches tensor operations
///
/// This function demonstrates and validates that our tensor operations are working correctly by:
/// 1. Extracting a specific value from the result of a matrix multiplication
/// 2. Manually calculating the same value using dot product
/// 3. Comparing the results to ensure they match
///
/// This is useful for:
/// - Debugging tensor operations
/// - Understanding how matrix multiplication works at a fundamental level
/// - Verifying that our neural network calculations are correct
///
/// # Arguments
/// * `xenc` - One-hot encoded input tensor
/// * `xenc_w` - Result of matrix multiplication between xenc and weights
/// * `w` - Weight matrix
/// * `row_idx` - Row index to check (input position)
/// * `col_idx` - Column index to check (neuron/output position)
///
/// # Returns
/// * Result containing the computed value
pub fn verify_matrix_multiplication(
    xenc: &Tensor,
    xenc_w: &Tensor,
    w: &Tensor,
    row_idx: usize,
    col_idx: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let value = xenc_w.i(row_idx)?.i(col_idx)?;

    let row = xenc.i(row_idx)?.to_vec1::<f32>()?;
    let col = w.i(col_idx)?.to_vec1::<f32>()?;

    let mut manual_dot = 0.0;
    for i in 0..27 {
        manual_dot += row[i] * col[i];
    }

    let tensor_value = value.to_scalar::<f32>()?;
    assert!(
        (manual_dot - tensor_value).abs() < 1e-5,
        "Manual calculation ({}) doesn't match tensor operation ({})",
        manual_dot,
        tensor_value
    );

    Ok(tensor_value)
}

/// Applies softmax activation to convert logits into probabilities. This is the forward pass.
///
/// The softmax function converts raw model outputs (logits) into probabilities by:
/// 1. Taking the exponential of each logit (to make all values positive)
/// 2. Normalizing by dividing by the sum of all exponentials
///
/// This transformation ensures:
/// - All outputs are between 0 and 1
/// - Outputs sum to 1 (making them valid probabilities)
/// - Preserves relative differences (larger logits -> larger probabilities)
///
/// The function follows this formula:
/// softmax(x_i) = exp(x_i) / Σ exp(x_j)
///
/// Why this works: we've created differentiable operations where the output can be
/// interpreted as probabilities. Our xenc @ w gives us logits, which we can interpret
/// as log counts, which we exponentiate to get counts, which we normalize to get a probability distribution.
///
/// We can then backpropagate through this process to update our weights and iterate
/// towards minimizing a loss function. As we tune the weights, we expect the model
/// to get better at predicting the next character in a sequence. Can we optimize and find a good W,
/// such that the probabilities coming out are pretty good?
///
/// # Arguments
/// * `logits` - Tensor of raw model outputs
/// * `device` - Device to store tensors on (CPU/GPU)
///
/// # Returns
/// * Tensor of probabilities
pub fn apply_softmax(logits: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Convert logits to exponential scale (all positive numbers)
    // Equivalent to N(w, x)
    let counts = logits.exp()?;

    // Sum along dimension 1, keeping dimensions for broadcasting
    let sum = counts.sum_keepdim(1)?;

    // Broadcast sum to match counts shape for element-wise division
    let sum_broadcast = sum.broadcast_as(counts.shape())?;

    // Normalize to get probabilities
    let prob = (counts / sum_broadcast)?;

    Ok(prob)
}
