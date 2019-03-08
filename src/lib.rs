use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn generate_sample_point(
    rng: &mut Xoshiro256Plus,
    i: f64,
    j: f64,
    xhalf: f64,
    yhalf: f64,
    n: f64,
) -> (f64, f64) {
    (
        (i + 0.5 * (xhalf + rng.gen::<f64>())) / n,
        (j + 0.5 * (yhalf + rng.gen::<f64>())) / n,
    )
}

fn extend_sequence(rng: &mut Xoshiro256Plus, samples: &mut Vec<(f64, f64)>, N: usize) {
    let n = (N as f64).sqrt();
    for s in 0..N {
        // determine subquadrant of existing sample point
        let oldpt = samples[s];
        let i = (n * oldpt.0).floor();
        let j = (n * oldpt.1).floor();
        let mut xhalf = (2.0 * (n * oldpt.0 - i)).floor();
        let mut yhalf = (2.0 * (n * oldpt.1 - j)).floor();

        // first select the diagonally opposite sub-quadrant
        xhalf = 1.0 - xhalf;
        yhalf = 1.0 - yhalf;

        samples[N + s] = generate_sample_point(rng, i, j, xhalf, yhalf, n);

        // then randomly select one of the two remaining subquadrants
        if rng.gen::<f64>() < 0.5 {
            xhalf = 1.0 - xhalf;
        } else {
            yhalf = 1.0 - yhalf;
        }
        samples[2 * N + s] = generate_sample_point(rng, i, j, xhalf, yhalf, n);

        // and finally the last one
        xhalf = 1.0 - xhalf;
        yhalf = 1.0 - yhalf;
        samples[3 * N + s] = generate_sample_point(rng, i, j, xhalf, yhalf, n);
    }
}

pub fn generate_pj(sqrt_num_samples: usize) -> Vec<(f64, f64)> {
    generate_pj_seeded(sqrt_num_samples, 0)
}

pub fn generate_pj_seeded(num_samples: usize, seed: u64) -> Vec<(f64, f64)> {
    let num_samples = num_samples.next_power_of_two(); // I love rust
    let mut samples = Vec::new();
    samples.resize(num_samples, (0.0, 0.0));
    // generate first sample at random position
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    samples[0] = (rng.gen::<f64>(), rng.gen::<f64>());

    let mut N = 1;
    while N < num_samples {
        // generate next 3N sample points
        extend_sequence(&mut rng, &mut samples, N);
        N = 4 * N;
    }

    samples
}

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log2(x: usize) -> u32 {
    assert!(x > 0);
    num_bits::<usize>() as u32 - x.leading_zeros() - 1
}

fn generate_sample_point_mj(
    rng: &mut Xoshiro256Plus,
    i: f64,
    j: f64,
    xhalf: f64,
    yhalf: f64,
    n: f64,
    N: f64,
    occupied_1d_x: &mut [bool],
    occupied_1d_y: &mut [bool],
    samples: &mut Vec<(f64, f64)>,
) {
    let NN = 2.0 * N;
    let mut best_dist = 0.0;
    let mut num_cand = 10;
    let mut cand_pt = (0.0, 0.0);
    // generate candidate points and pick the best
    for t in 1..num_cand {
        // generate candidate x coord
        loop {
            cand_pt.0 = (i + 0.5 * (xhalf + rng.gen::<f64>())) / n;
            if !occupied_1d_x[(NN * cand_pt.0).floor() as usize] {
                break;
            }
        }
        // generate candidate y coord
        loop {
            cand_pt.1 = (j + 0.5 * (yhalf + rng.gen::<f64>())) / n;
            if !occupied_1d_y[(NN * cand_pt.1).floor() as usize] {
                break;
            }
        }
        // evaluate distance between candidate point and existing
        // samples for blue noise properties
        // TODO
    }
    // mark 1d strata as occupied
    occupied_1d_x[(NN * cand_pt.0).floor() as usize] = true;
    occupied_1d_y[(NN * cand_pt.1).floor() as usize] = true;
    // assign new sample point
    samples.push(cand_pt);
}

fn mark_occupied_strata_pmj(
    N: usize,
    occupied_1d_x: &mut [bool],
    occupied_1d_y: &mut [bool],
    samples: &[(f64, f64)],
) {
    let NN = 2 * N;
    for x in occupied_1d_x.iter_mut().take(NN) {
        *x = false;
    }
    for y in occupied_1d_y.iter_mut().take(NN) {
        *y = false;
    }

    for s in 0..N {
        let xs = (NN as f64 * samples[s].0).floor() as usize;
        let ys = (NN as f64 * samples[s].1).floor() as usize;
        occupied_1d_x[xs] = true;
        occupied_1d_y[ys] = true;
    }
}

fn extend_sequence_even(
    rng: &mut Xoshiro256Plus,
    N: f64,
    occupied_1d_x: &mut [bool],
    occupied_1d_y: &mut [bool],
    samples: &mut Vec<(f64, f64)>,
) {
    let n = N.sqrt();
    // mark already occupied 1d strata so we can avoid them
    mark_occupied_strata_pmj(N as usize, occupied_1d_x, occupied_1d_y, samples.as_slice());
    // loop over N old samples and generate one new sample for each
    for s in 0..(N as usize) {
        let oldpt = samples[s];
        let i = (n * oldpt.0).floor();
        let j = (n * oldpt.1).floor();
        let xhalf = (2.0 * (n * oldpt.0 - i)).floor();
        let yhalf = (2.0 * (n * oldpt.1 - j)).floor();
        // select the diagonally opposite subquadrant
        let xhalf = 1.0 - xhalf;
        let yhalf = 1.0 - yhalf;
        // generate a sample point
        generate_sample_point_mj(
            rng,
            i,
            j,
            xhalf,
            yhalf,
            n,
            N,
            occupied_1d_x,
            occupied_1d_y,
            samples,
        );
    }
}

fn extend_sequence_odd(
    rng: &mut Xoshiro256Plus,
    N: f64,
    occupied_1d_x: &mut [bool],
    occupied_1d_y: &mut [bool],
    xhalves: &mut [f64],
    yhalves: &mut [f64],
    samples: &mut Vec<(f64, f64)>,
) {
    let n = N.sqrt();
    // mark already occupied 1d strata so we can avoid them
    mark_occupied_strata_pmj(N as usize, occupied_1d_x, occupied_1d_y, samples.as_slice());
    // (Optionally:
    // 1) Classify occupied sub-pixels: odd or even diagonal
    // 2) Pre-select well-balanced subquadrants here for better
    // sample distribution between powers of two samples)
    // Loop over N/2 old samples and generate 2 new samples for each // – one at a time to keep the order consecutive (for "greedy"
    // best candidates)
    // Select one of the two remaining subquadrants
    for s in 0..(N as usize / 2) {
        let oldpt = samples[s];
        let i = (n * oldpt.0).floor();
        let j = (n * oldpt.1).floor();
        let mut xhalf = (2.0 * (n * oldpt.0 - i)).floor();
        let mut yhalf = (2.0 * (n * oldpt.1 - j)).floor();
        // randomly select one of the two remaining subquadrants
        if rng.gen::<f64>() < 0.5 {
            xhalf = 1.0 - xhalf;
        } else {
            yhalf = 1.0 - yhalf;
        }
        xhalves[s] = xhalf;
        yhalves[s] = yhalf;
        // generate a sample point
        generate_sample_point_mj(
            rng,
            i,
            j,
            xhalf,
            yhalf,
            n,
            N,
            occupied_1d_x,
            occupied_1d_y,
            samples,
        );
    }
    // and finally fill in the last subquadrants
    for s in 0..(N as usize / 2) {
        let oldpt = samples[s];
        let i = (n * oldpt.0).floor();
        let j = (n * oldpt.1).floor();
        let xhalf = 1.0 - xhalves[s];
        let yhalf = 1.0 - yhalves[s];
        // generate a sample point
        generate_sample_point_mj(
            rng,
            i,
            j,
            xhalf,
            yhalf,
            n,
            N,
            occupied_1d_x,
            occupied_1d_y,
            samples,
        );
    }
}

pub fn generate_pmj_seeded(num_samples: usize, seed: u64) -> Vec<(f64, f64)> {
    let num_samples = num_samples.next_power_of_two(); // I love rust
    let mut samples = vec![(0.0, 0.0); num_samples];
    // generate first sample at random position
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    samples[0] = (rng.gen::<f64>(), rng.gen::<f64>());

    let mut occupied_1d_x = vec![false; num_samples];
    let mut occupied_1d_y = vec![false; num_samples];

    let mut xhalves = vec![0.0; num_samples];
    let mut yhalves = vec![0.0; num_samples];

    let mut N = 1;
    while N < num_samples {
        // generate next 3N sample points
        if log2(N) % 2 == 0 {
            // even power of two
            extend_sequence_even(
                &mut rng,
                N as f64,
                &mut occupied_1d_x,
                &mut occupied_1d_y,
                &mut samples,
            );
        } else {
            // odd power of two
            extend_sequence_odd(
                &mut rng,
                N as f64,
                &mut occupied_1d_x,
                &mut occupied_1d_y,
                &mut xhalves,
                &mut yhalves,
                &mut samples,
            );
        }

        N *= 4;
    }

    samples
}

#[cfg(test)]
#[test]
fn it_works() {
    let samples = generate_pj(2);
    assert_eq!(
        samples,
        [
            (0.8541927863674711, 0.19272815297677148),
            (0.48774904600841795, 0.6558958405065498),
            (0.5072163698519102, 0.8480898038405176),
            (0.03321444179621752, 0.3255121930060155),
        ]
    )
}
