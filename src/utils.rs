use anyhow::Result;
use candle_core::Tensor;
use std::collections::HashMap;

pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

pub fn tensor_to_bigram_hashmap(
    tensor: &Tensor,
    chars: &[String],
) -> Result<HashMap<(String, String), f64>> {
    let data = tensor.to_vec2::<f64>()?;
    let mut bigram_map = HashMap::new();

    for (i, row) in data.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value > 0.0 {
                bigram_map.insert((chars[i].clone(), chars[j].clone()), value);
            }
        }
    }

    Ok(bigram_map)
}
