use anyhow::Result;
use candle_core::{Device, Tensor};
use makemore_rs::{apply_softmax, create_character_pairs, create_one_hot_encoding};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let words = vec!["emma".to_string()];
    let (xs, _) = create_character_pairs(&words).map_err(|e| anyhow::anyhow!("{}", e))?;

    let xs_tensor = Tensor::new(xs, &device).map_err(|e| anyhow::anyhow!("{}", e))?;
    let xenc =
        create_one_hot_encoding(&xs_tensor, 27, &device).map_err(|e| anyhow::anyhow!("{}", e))?;
    let w = Tensor::randn(0f32, 1f32, (27, 27), &device).map_err(|e| anyhow::anyhow!("{}", e))?;

    let logits = xenc.matmul(&w).map_err(|e| anyhow::anyhow!("{}", e))?;
    let probs = apply_softmax(&logits).map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("probs shape: {:?}", probs.shape());
    println!("probs: {:?}", probs.to_vec2::<f32>()?);
    println!("probs sum: {:?}", probs.sum(1)?.to_vec1::<f32>()?);

    Ok(())
}
