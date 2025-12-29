use std::path::PathBuf;

use clap::Parser;

mod pipeline;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long, default_value = "rust_cache")]
    output_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    write_intermediate: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut config = pipeline::load_config(&args.config)?;
    pipeline::resolve_data_dir(&mut config, &args.config)?;
    pipeline::run_pipeline(&config, &args.output_dir, args.write_intermediate)?;
    Ok(())
}
