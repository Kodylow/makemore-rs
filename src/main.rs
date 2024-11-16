use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::{Optimizer, SGD};
use makemore_rs::{apply_softmax, create_character_pairs, create_one_hot_encoding, index_to_char};
use rand::distributions::Distribution;

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
    // Load training data
    let names = makemore_rs::data::load_names("./names.txt");

    // Convert names to Strings first
    let names: Vec<String> = names.iter().map(|n| n.name.clone()).collect();
    println!("Unique names: {}", names.len());
    let (xs, ys) = create_character_pairs(&names)?;
    println!("xs length: {:?}", xs.len());
    println!("ys length: {:?}", ys.len());
    let device = Device::Cpu;

    // Convert to tensors for GPU/CPU acceleration
    let xs_tensor = Tensor::new(xs, &device)?;
    let ys_tensor = Tensor::new(ys, &device)?;

    // Initialize weight matrix with random values
    // Using Var instead of Tensor enables automatic gradient tracking
    // Shape is (27,27) for transitions between all possible characters
    let w = Var::randn(0.0, 1.0, (27, 27), &device)?;

    // Create SGD optimizer to update weights
    // Learning rate 50.0 controls size of weight updates
    // Big for this simple model
    let mut opt = SGD::new(vec![w.clone()], 50.0)?;

    // Training loop - each iteration:
    // 1. Forward pass to get predictions
    // 2. Calculate loss
    // 3. Backprop gradients
    // 4. Update weights
    for k in 0..10 {
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

        // Calculate negative log likelihood loss using batch operations
        let indices = Tensor::arange(0, xs_tensor.dims()[0] as i64, &device)?;
        let target_probs = probs
            .index_select(&indices, 0)?
            .gather(&ys_tensor.unsqueeze(1)?, 1)?
            .squeeze(1)?;
        let loss = target_probs.log()?.neg()?.mean_all()?;

        // Add L2 regularization like in the Python version
        let l2_loss = w_f32
            .powf(2.0)?
            .mean_all()?
            .mul(&Tensor::new(0.01f32, &device)?)?;
        let loss = loss.add(&l2_loss)?;

        println!("Step {}, Loss: {}", k, loss.to_scalar::<f32>()?);

        // Compute gradients and update weights with SGD
        opt.backward_step(&loss)?;
    }

    // Generation loop
    let mut rng = rand::thread_rng();
    for _ in 0..5 {
        let mut out = Vec::new();
        let mut ix = 0; // Start with first character (.)

        loop {
            // Convert current character index to one-hot
            let x_tensor = Tensor::new(&[ix as i64], &device)?;
            let xenc = create_one_hot_encoding(&x_tensor, 27, &device)?.to_dtype(DType::F32)?;

            // Get probabilities for next character
            let logits = xenc.matmul(&w.to_dtype(DType::F32)?)?;
            let probs = apply_softmax(&logits)?;

            // Sample from probability distribution
            // Squeeze to remove the extra dimension [1, 27] -> [27]
            let prob_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;
            let dist = rand::distributions::WeightedIndex::new(&prob_vec)?;
            ix = dist.sample(&mut rng);

            // Convert index back to character and append
            let c = index_to_char(ix);
            out.push(c);

            // Break if we generated end token or name is too long
            if ix == 0 || out.len() > 20 {
                break;
            }
        }

        // Print generated name
        println!("Generated: {}", out.into_iter().collect::<String>());
    }

    Ok(())
}
