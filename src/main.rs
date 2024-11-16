use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::{Optimizer, SGD};
use makemore_rs::{apply_softmax, create_character_pairs, create_one_hot_encoding};

/// Trains a simple character-level language model using stochastic gradient descent
///
/// This example demonstrates building a neural network that learns to predict the next character
/// in a sequence. It implements a basic bigram model that captures character transition probabilities.
///
/// The model architecture:
/// 1. Input layer: One-hot encoded characters (27 dimensions for a-z + '.')
/// 2. Weight matrix: 27x27 learnable parameters
/// 3. Output layer: Softmax probabilities over next character
///
/// Training process:
/// - Forward pass: Convert input -> probabilities
/// - Loss calculation: Negative log likelihood of true next char
/// - Backward pass: Compute gradients
/// - Parameter update: SGD step to minimize loss
///
/// Key concepts demonstrated:
/// - Working with tensors and automatic differentiation
/// - Converting discrete tokens to continuous vectors
/// - Maximum likelihood training with cross-entropy loss
/// - Gradient-based optimization
///
/// # Returns
/// * Result indicating success or error during training
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load training data - here just using "emma" as example
    let words = vec!["emma".to_string()];

    // Convert characters to training pairs (input char -> target char)
    let (xs, ys) = create_character_pairs(&words)?;
    let device = Device::Cpu;

    // Convert to tensors for GPU/CPU acceleration
    let xs_tensor = Tensor::new(xs, &device)?;
    let ys_tensor = Tensor::new(ys, &device)?;

    // Initialize weight matrix with random values
    // Using Var instead of Tensor enables automatic gradient tracking
    // Shape is (27,27) for transitions between all possible characters
    let w = Var::randn(0.0, 1.0, (27, 27), &device)?;
    println!("Weight dtype: {:?}", w.dtype());

    // Create SGD optimizer to update weights
    // Learning rate 0.1 controls size of weight updates
    let mut opt = SGD::new(vec![w.clone()], 0.1)?;

    // Training loop - each iteration:
    // 1. Forward pass to get predictions
    // 2. Calculate loss
    // 3. Backprop gradients
    // 4. Update weights
    for k in 0..1000 {
        // Convert input chars to one-hot vectors
        // This creates a sparse binary matrix where each row has a single 1
        let xenc = create_one_hot_encoding(&xs_tensor, 27, &device)?.to_dtype(DType::F32)?;

        // Ensure weights are f32 for matmul
        let w_f32 = w.to_dtype(DType::F32)?;

        // Forward pass: multiply one-hot vectors by weights
        // This computes raw logit scores for each possible next character
        let logits = xenc.matmul(&w_f32)?;

        // Convert logits to probabilities with softmax
        // Now each row sums to 1 and represents a probability distribution
        let probs = apply_softmax(&logits)?;

        // Calculate negative log likelihood loss
        // This measures how well we predict the true next character
        let mut loss = Tensor::zeros((), DType::F32, &device)?;
        for i in 0..5 {
            // Get true next char index and its predicted probability
            let y = ys_tensor.i(i)?.to_scalar::<i64>()? as usize;
            let p = probs.i(i)?.i(y)?;
            // Add negative log prob to loss
            loss = loss.add(&p.log()?.neg()?)?;
        }
        // Average loss over sequence length
        let loss = loss.div(&Tensor::new(5.0f32, &device)?)?;

        println!("Step {}, Loss: {}", k, loss.to_scalar::<f32>()?);

        // Compute gradients and update weights with SGD
        opt.backward_step(&loss)?;
    }

    Ok(())
}
