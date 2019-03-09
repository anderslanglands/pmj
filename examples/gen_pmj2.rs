use pmj::*;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name="pmj02", about="Generate a pmj02 sequence")]
struct Opt {
    #[structopt(short="n", long="numsamples", default_value="1024")]
    numsamples: usize,
    #[structopt(short="s", long="seed", default_value="0")]
    seed: u64,
}

pub fn main() {
    let opt = Opt::from_args();
    let samples = generate_pmj_seeded2(opt.numsamples, opt.seed);
    println!("# Generated samples:" );
    for s in samples.iter().take(opt.numsamples) {
        println!("{} {}", s.0, s.1);
    }
}