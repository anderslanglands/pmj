use pmj::*;

pub fn main() {
    let samples = generate_pmj02(1024);
    println!("# Generated samples:" );
    for (i, s) in samples.iter().enumerate() {
        println!("{} {}", s.0, s.1);
    }
}