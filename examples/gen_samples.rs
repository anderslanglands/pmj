use pmj::*;

pub fn main() {
    let samples = generate_pmj_seeded(16, 0);
    for s in samples {
        println!("{} {}", s.0, s.1);
    }
}