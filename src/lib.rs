pub mod bigrams;
pub mod data;
pub mod plot;
pub mod vocabulary;

pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}
