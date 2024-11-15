//! This example demonstrates using log probabilities to evaluate a bigram language model.
//!
//! Log probabilities are crucial in machine learning and language modeling for several reasons:
//!
//! 1. Numerical Stability: When multiplying many small probabilities (as in language models),
//!    the result can become too small for floating point precision. Log probabilities turn
//!    multiplications into additions, preventing underflow.
//!
//! 2. Model Quality Assessment: Log probabilities help assess how "surprised" the model is
//!    by sequences. More negative values mean the model considers that sequence unlikely.
//!    Better models should assign higher (less negative) log probabilities to valid sequences.
//!
//! 3. Training Objective: The negative log probability is commonly used as a loss function
//!    in language models, where minimizing it maximizes the likelihood of the training data.
//!
//! This example generates names using a bigram model and shows both raw probabilities and
//! their logarithms. The log probabilities of each character transition indicate how natural
//! or unusual that character combination is according to the training data.

use anyhow::Result;
use candle_core::Device;
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use tracing::info;

fn main() -> Result<()> {
    makemore_rs::utils::init_logging();
    let device = Device::Cpu;

    let names = load_names_unique("./names.txt");
    let model = BigramModel::new(&names, &device)?;

    info!("Generating names with bigram probabilities:");
    for _ in 0..5 {
        let mut name = Vec::new();
        #[allow(unused_assignments)]
        let mut ix = 0;
        let mut prev_ix = 0;
        let mut log_likelihood = 0.0;

        info!("New name:");
        loop {
            let probs = model.get_probabilities();
            ix = model.multinomial(&probs, 1, true)?.to_vec1::<i64>()?[0] as usize
                % model.get_vocabulary().get_size();

            // Get the characters and probability
            let ch1 = model.get_vocabulary().get_char(prev_ix);
            let ch2 = model.get_vocabulary().get_char(ix);
            let prob = probs.get(prev_ix)?.get(ix)?;
            let logprob = prob.log()?;
            log_likelihood += logprob.to_vec0::<f32>()?;
            // Print the bigram and its probabilities
            info!(
                "  {}{}: prob={:.4}, logprob={:.4}",
                ch1,
                ch2,
                prob.to_vec0::<f32>()?,
                logprob.to_vec0::<f32>()?
            );

            name.push(ch2);

            if ix == 0 {
                break;
            }
            prev_ix = ix;
        }

        info!(
            "Generated: {}",
            name.iter().map(|c| c.as_str()).collect::<String>()
        );
        info!("Log likelihood: {}", log_likelihood);
        info!("---");
    }

    Ok(())
}
