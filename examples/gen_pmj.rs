use pmj::*;

pub fn main() {
    let samples = generate_pmj_seeded(1024, 0);
    println!("# Generated samples:" );
    for s in samples {
        println!("{} {}", s.0, s.1);
    }
}