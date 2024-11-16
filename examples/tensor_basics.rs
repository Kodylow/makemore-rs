use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;
use makemore_rs::create_character_pairs;
use makemore_rs::create_one_hot_encoding;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let words = vec!["emma".to_string()];
    let (xs, ys) = create_character_pairs(&words).map_err(|e| anyhow::anyhow!("{}", e))?;

    let xs_tensor = Tensor::new(xs, &device).map_err(|e| anyhow::anyhow!("{}", e))?;
    let ys_tensor = Tensor::new(ys, &device).map_err(|e| anyhow::anyhow!("{}", e))?;
    let xenc =
        create_one_hot_encoding(&xs_tensor, 27, &device).map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("xs: {:?}", xs_tensor);
    println!("ys: {:?}", ys_tensor);
    println!("xenc: {:?}", xenc);

    Ok(())
}
