use anyhow::Result;
use candle_core::Device;
use makemore_rs::bigrams::BigramModel;
use makemore_rs::data::load_names_unique;
use tracing::info;

fn main() -> Result<()> {
    makemore_rs::utils::init_logging();
    let device = Device::Cpu;

    // Create and train the model
    let names = load_names_unique("./names.txt");
    let model = BigramModel::new(&names, &device)?;

    // Generate 5 names
    info!("Generating names:");
    for _ in 0..5 {
        let mut name = Vec::new();
        #[allow(unused_assignments)]
        let mut ix = 0;

        loop {
            let probs = model.get_probabilities();
            ix = model.multinomial(&probs, 1, true)?.to_vec1::<i64>()?[0] as usize
                % model.get_vocabulary().get_size();
            name.push(model.get_vocabulary().get_char(ix));

            if ix == 0 {
                break;
            }
        }

        info!("{}", name.iter().map(|c| c.as_str()).collect::<String>());
    }

    Ok(())
}
