#![allow(non_snake_case)]

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
    let sqrt_num_samples = ((num_samples as f64).sqrt() as usize).next_power_of_two();
    let num_samples = sqrt_num_samples * sqrt_num_samples;
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
    // println!("\t\t\tGenerating sample point with:");
    // println!("\t\t\ti: {}", i);
    // println!("\t\t\tj: {}", j);
    // println!("\t\t\txhalf: {}", xhalf);
    // println!("\t\t\tyhalf: {}", yhalf);
    // println!("\t\t\tn: {}", n);
    // println!("\t\t\tN: {}", N);
    // println!("\t\t\tocc_x: {:?}", occupied_1d_x);
    // println!("\t\t\tocc_y: {:?}", occupied_1d_y);
    
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
    // println!("Generated ({:?})", cand_pt);
    // mark 1d strata as occupied
    // println!("Marking x[{}]", (NN * cand_pt.0).floor() as usize);
    occupied_1d_x[(NN * cand_pt.0).floor() as usize] = true;
    // println!("Marking y[{}]", (NN * cand_pt.1).floor() as usize);
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
    // println!("Extending even");
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

fn classify_sub_quadrants(
    n: f64, 
    N: f64, 
    samples: &[(f64, f64)], 
    evendiags: &mut Vec<Vec<bool>>
) {
    let nn = 2.0 * n;
    for s in 0..((N/2.0) as usize) {
        let xstratum = (nn * samples[s].0).floor() as usize;
        let ystratum = (nn * samples[s].1).floor() as usize;
        let evenx = xstratum % 2;
        let eveny = ystratum % 2;
        evendiags[ystratum/2][xstratum/2] = (evenx == eveny);
    }
}

fn select_sub_quadrants(
    rng: &mut Xoshiro256Plus, 
    n: usize, 
    choice_balance_x: &mut[i64], 
    choice_balance_y: &mut[i64], 
    evendiags: &mut Vec<Vec<bool>>,
    subquadchoices_x: &mut Vec<Vec<usize>>,
    subquadchoices_y: &mut Vec<Vec<usize>>,
) -> bool {
    for x in choice_balance_x.iter_mut().take(n) {
        *x = 0;
    }
    for y in choice_balance_y.iter_mut().take(n) {
        *y = 0;
    }
    let mut up = false;
    // visit quadrants in up/down "ox-plowing" order
    for i in 0..n {
        up = !up;
        for jj in 0..n {
            let j = if up {jj} else {n - jj - 1};
            let last = ((jj as i64) == ((n as i64) - choice_balance_x[i].abs())) && (n > 1);
            let evendiag = evendiags[j][i];
            // if last entry in a column, enforce balance
            if choice_balance_y[j] != 0 && !last {
                let neg = choice_balance_y[j] < 0; // more y lows than highs
                // do opposite y choice than previous column
                subquadchoices_y[j][i] = if neg {1} else {0};
                subquadchoices_x[j][i] = if evendiag ^ neg {1} else {0};
                choice_balance_y[j] += if neg {1} else {-1};
                choice_balance_x[i] += if evendiag ^ neg {1} else {-1};
            } else if choice_balance_x[i] != 0 {
                let neg = choice_balance_x[i] < 0; // more x lows than highs
                subquadchoices_x[j][i] = if neg {1} else {0};
                subquadchoices_y[j][i] = if evendiag ^ neg {1} else {0};
                choice_balance_x[i] += if neg {1} else {-1};
                choice_balance_y[j] += if evendiag ^ neg {1} else {-1};
            } else {
                // even balance in both x and y, randomly select one subquadrant
                let xhalf = if rng.gen::<f64>() > 0.5 {1} else {0};
                let yhalf = if evendiag {1 - xhalf} else {xhalf};
                subquadchoices_x[j][i] = xhalf;
                subquadchoices_y[j][i] = yhalf;
                choice_balance_x[i] += if xhalf != 0 {1} else {-1};
                choice_balance_y[j] += if yhalf != 0 {1} else {-1};
            }
        }
    }
    if n == 1 {
        return true;
    }
    for y in choice_balance_y.iter().take(n) {
        if *y != 0 {
            return false;
        }
    }

    true
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
    // println!("Extending odd");
    let n = (N/2.0).sqrt();
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

fn extend_sequence_odd2(
    rng: &mut Xoshiro256Plus,
    N: f64,
    occupied_1d_x: &mut [bool],
    occupied_1d_y: &mut [bool],
    xhalves: &mut [f64],
    yhalves: &mut [f64],
    samples: &mut Vec<(f64, f64)>,
    choice_balance_x: &mut[i64], 
    choice_balance_y: &mut[i64], 
    evendiags: &mut Vec<Vec<bool>>,
    subquadchoices_x: &mut Vec<Vec<usize>>,
    subquadchoices_y: &mut Vec<Vec<usize>>,
) {
    // println!("Extending odd");
    let n = (N/2.0).sqrt();
    // mark already occupied 1d strata so we can avoid them
    mark_occupied_strata_pmj(N as usize, occupied_1d_x, occupied_1d_y, samples.as_slice());
    // (Optionally:
    // 1) Classify occupied sub-pixels: odd or even diagonal
    classify_sub_quadrants(n, N, samples, evendiags);
    // 2) Pre-select well-balanced subquadrants here for better
    // sample distribution between powers of two samples)
    loop {
        if select_sub_quadrants(rng, n as usize, choice_balance_x, choice_balance_y, evendiags, subquadchoices_x, subquadchoices_y) {
            break;
        }
    }
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
    let sqrt_num_samples = ((num_samples as f64).sqrt() as usize).next_power_of_two();
    let num_samples = sqrt_num_samples * sqrt_num_samples;
    let mut samples = Vec::new();
    // generate first sample at random position
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    samples.push((rng.gen::<f64>(), rng.gen::<f64>()));

    let mut occupied_1d_x = vec![false; num_samples];
    let mut occupied_1d_y = vec![false; num_samples];

    let mut xhalves = vec![0.0; num_samples];
    let mut yhalves = vec![0.0; num_samples];

    let mut N = 1;
    while N < num_samples {
        // generate next 3N sample points
        extend_sequence_even(
            &mut rng,
            N as f64,
            &mut occupied_1d_x,
            &mut occupied_1d_y,
            &mut samples,
        );
        extend_sequence_odd(
            &mut rng,
            2.0 * N as f64,
            &mut occupied_1d_x,
            &mut occupied_1d_y,
            &mut xhalves,
            &mut yhalves,
            &mut samples,
        );

        N *= 4;
    }

    samples
}

pub fn generate_pmj_seeded2(num_samples: usize, seed: u64) -> Vec<(f64, f64)> {
    let sqrt_num_samples = ((num_samples as f64).sqrt() as usize).next_power_of_two();
    let num_samples = sqrt_num_samples * sqrt_num_samples;
    let mut samples = Vec::new();
    // generate first sample at random position
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    samples.push((rng.gen::<f64>(), rng.gen::<f64>()));

    let mut occupied_1d_x = vec![false; num_samples];
    let mut occupied_1d_y = vec![false; num_samples];

    let mut xhalves = vec![0.0; num_samples];
    let mut yhalves = vec![0.0; num_samples];

    let mut choice_balance_x = vec![0i64; num_samples];
    let mut choice_balance_y = vec![0i64; num_samples];
    
    let mut evendiags = vec![vec![false; num_samples]; num_samples]; 

    let mut subquadchoices_x = vec![vec![0usize; num_samples]; num_samples];
    let mut subquadchoices_y = vec![vec![0usize; num_samples]; num_samples];

    let mut N = 1;
    while N < num_samples {
        // generate next 3N sample points
        extend_sequence_even(
            &mut rng,
            N as f64,
            &mut occupied_1d_x,
            &mut occupied_1d_y,
            &mut samples,
        );
        extend_sequence_odd2(
            &mut rng,
            2.0 * N as f64,
            &mut occupied_1d_x,
            &mut occupied_1d_y,
            &mut xhalves,
            &mut yhalves,
            &mut samples,
            &mut choice_balance_x,
            &mut choice_balance_y,
            &mut evendiags,
            &mut subquadchoices_x,
            &mut subquadchoices_y,
        );

        N *= 4;
    }

    samples
}

pub fn generate_pmj(num_samples: usize) -> Vec<(f64, f64)> {
    generate_pmj_seeded(num_samples, 0)
}

pub fn generate_pmj2(num_samples: usize) -> Vec<(f64, f64)> {
    generate_pmj_seeded2(num_samples, 0)
}

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log2(x: usize) -> u32 {
    assert!(x > 0);
    num_bits::<usize>() as u32 - x.leading_zeros() - 1
}

fn mark_occupied_strata_pmj02(
    N: usize, 
    occupied_strata: &mut Vec<Vec<bool>>,
    samples: &[(f64, f64)],
) {
    let NN = 2 * N;
    for row in occupied_strata.iter_mut().take(log2(NN) as usize) {
        for i in row.iter_mut().take(NN) {
            *i = false;
        }
    }
    for s in samples.iter().take(N) {
        mark_occupied_strata_1(*s, NN, occupied_strata);
    }
}

fn mark_occupied_strata_1(
    pt: (f64, f64), 
    NN: usize, 
    occupied_strata: &mut Vec<Vec<bool>>
) {
    let mut shape = 0;
    let mut xdivs = NN;
    let mut ydivs = 1;
    while xdivs != 0 {
        let xstratum = (xdivs as f64 * pt.0).floor() as usize;
        let ystratum = (ydivs as f64 * pt.1).floor() as usize;
        occupied_strata[shape][ystratum * xdivs + xstratum] = true;
        shape += 1;
        xdivs /= 2;
        ydivs *= 2;
    }
}

fn generate_sample_point_pmj02(
    rng: &mut Xoshiro256Plus,
    i: f64,
    j: f64,
    xhalf: f64,
    yhalf: f64,
    n: f64,
    N: f64,
    occupied_strata: &mut Vec<Vec<bool>>,
    samples: &mut Vec<(f64, f64)>,
) {
    let NN = 2.0 * N;
    // Generate x and y until sample is accepted as an (0,2) sample
    let mut pt = (0.0f64, 0.0f64);
    loop {
        pt = (
            (i + 0.5 * (xhalf + rng.gen::<f64>())) / n,
            (j + 0.5 * (yhalf + rng.gen::<f64>())) / n,
        );
        if !is_occupied(pt, NN as usize, occupied_strata) {
            break;
        }
    }
    // mark strata that this new point occupies
    mark_occupied_strata_1(pt, NN as usize, occupied_strata);
    samples.push(pt);
}

fn is_occupied(pt: (f64, f64), NN: usize, occupied_strata: &Vec<Vec<bool>>) -> bool {
    let mut shape = 0;
    let mut xdivs = NN;
    let mut ydivs = 1;
    while xdivs != 0 {
        let xstratum = (xdivs as f64 * pt.0).floor() as usize;
        let ystratum = (ydivs as f64 * pt.1).floor() as usize;
        if occupied_strata[shape][ystratum*xdivs+xstratum] {
            return true;
        }
        shape += 1;
        xdivs /= 2;
        ydivs *= 2;
    }

    false
}

fn extend_sequence_even_pmj02(
    rng: &mut Xoshiro256Plus,
    N: f64,
    occupied_strata: &mut Vec<Vec<bool>>,
    samples: &mut Vec<(f64, f64)>,
) {
    // println!("Extending even");
    let n = N.sqrt();
    // mark already occupied 1d strata so we can avoid them
    mark_occupied_strata_pmj02(N as usize, occupied_strata, samples.as_slice());
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
        generate_sample_point_pmj02(
            rng,
            i,
            j,
            xhalf,
            yhalf,
            n,
            N,
            occupied_strata,
            samples,
        );
    }
}

fn extend_sequence_odd_pmj02(
    rng: &mut Xoshiro256Plus,
    N: f64,
    occupied_strata: &mut Vec<Vec<bool>>,
    xhalves: &mut [f64],
    yhalves: &mut [f64],
    samples: &mut Vec<(f64, f64)>,
    choice_balance_x: &mut[i64], 
    choice_balance_y: &mut[i64], 
    evendiags: &mut Vec<Vec<bool>>,
    subquadchoices_x: &mut Vec<Vec<usize>>,
    subquadchoices_y: &mut Vec<Vec<usize>>,
) {
    // println!("Extending odd");
    let n = (N/2.0).sqrt();
    // mark already occupied 1d strata so we can avoid them
    mark_occupied_strata_pmj02(N as usize, occupied_strata, samples.as_slice());
    // (Optionally:
    // 1) Classify occupied sub-pixels: odd or even diagonal
    classify_sub_quadrants(n, N, samples, evendiags);
    // 2) Pre-select well-balanced subquadrants here for better
    // sample distribution between powers of two samples)
    // loop {
    //     if select_sub_quadrants(rng, n as usize, choice_balance_x, choice_balance_y, evendiags, subquadchoices_x, subquadchoices_y) {
    //         break;
    //     }
    // }
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
        generate_sample_point_pmj02(
            rng,
            i,
            j,
            xhalf,
            yhalf,
            n,
            N,
            occupied_strata,
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
        generate_sample_point_pmj02(
            rng,
            i,
            j,
            xhalf,
            yhalf,
            n,
            N,
            occupied_strata,
            samples,
        );
    }
}

pub fn generate_pmj_seeded_02(num_samples: usize, seed: u64) -> Vec<(f64, f64)> {
    let sqrt_num_samples = ((num_samples as f64).sqrt() as usize).next_power_of_two();
    let num_samples = sqrt_num_samples * sqrt_num_samples;
    let mut samples = Vec::new();
    // generate first sample at random position
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);
    samples.push((rng.gen::<f64>(), rng.gen::<f64>()));

    let mut occupied_strata = vec![vec![false; num_samples]; num_samples];

    let mut xhalves = vec![0.0; num_samples];
    let mut yhalves = vec![0.0; num_samples];

    let mut choice_balance_x = vec![0i64; num_samples];
    let mut choice_balance_y = vec![0i64; num_samples];
    
    let mut evendiags = vec![vec![false; num_samples]; num_samples]; 

    let mut subquadchoices_x = vec![vec![0usize; num_samples]; num_samples];
    let mut subquadchoices_y = vec![vec![0usize; num_samples]; num_samples];

    let mut N = 1;
    while N < num_samples {
        // generate next 3N sample points
        extend_sequence_even_pmj02(
            &mut rng,
            N as f64,
            &mut occupied_strata,
            &mut samples,
        );
        extend_sequence_odd_pmj02(
            &mut rng,
            2.0 * N as f64,
            &mut occupied_strata,
            &mut xhalves,
            &mut yhalves,
            &mut samples,
            &mut choice_balance_x,
            &mut choice_balance_y,
            &mut evendiags,
            &mut subquadchoices_x,
            &mut subquadchoices_y,
        );

        N *= 4;
    }

    samples
}

pub fn generate_pmj02(num_samples: usize) -> Vec<(f64, f64)> {
    generate_pmj_seeded_02(num_samples, 0)
}


#[cfg(test)]
#[test]
fn test_generate_pj() {
    let samples = generate_pj(4);
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

#[cfg(test)]
#[test]
fn test_generate_pmj() {
    let samples = generate_pmj(4);
    assert_eq!(
        samples,
        [
            (0.8541927863674711, 0.19272815297677148),
            (0.24684294195541084, 0.538566623107706),
            (0.5955176479006135, 0.8436799993413805),
            (0.432043726788904, 0.3624930244074985),
        ]
    )
}
