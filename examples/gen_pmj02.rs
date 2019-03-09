use pmj::*;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name="pmj02", about="Generate a progressive multi-jittered (0,2) sequence")]
struct Opt {
    #[structopt(short="n", long="numsamples", default_value="1024")]
    numsamples: usize,
    #[structopt(short="s", long="seed", default_value="0")]
    seed: u64,
    #[structopt(short="q", long="selectquadrants")]
    select_quadrants: bool,
}

pub fn main() {
    let opt = Opt::from_args();
    let samples = generate_pmj_seeded_02(opt.numsamples, opt.seed, opt.select_quadrants);
    println!("# Generated {} samples:", samples.len());
    for s in samples.iter().take(opt.numsamples) {
        println!("{} {}", s.0, s.1);
    }
}