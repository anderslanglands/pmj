use pmj::*;

pub fn main() {
    let samples = generate_pmj(1024);
    println!("# Generated samples:" );
    for s in samples {
        println!("{} {}", s.0, s.1);
    }
}