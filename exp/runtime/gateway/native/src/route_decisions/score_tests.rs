//! Score-consistency envelope and diagnosable rejection regressions.

use super::tests::score_payload;
use super::*;

#[test]
fn a_finer_score_is_judged_against_the_latent_expectation_envelope() {
    // Thirds published as 0.33 each beside the exact latent expectation of
    // (0.333, 0.333, 0.334): the recomputed 0.99 differs by 0.011, yet one
    // unit distribution rounding to every bin explains the score exactly.
    let (admitted, provider) = score_payload(&[0.33, 0.33, 0.33], 1.001);
    public_decisions(provider, &admitted).expect("latent expectation of rounded bins");
    let thirds = [0.33, 0.33, 0.33];
    // Latent expectations of bins rounding to 0.33 span [0.995, 1.005]; a
    // half-thousandth rounding allowance closes the envelope.
    assert!(score_matches_distribution(0.995, &thirds));
    assert!(score_matches_distribution(1.005, &thirds));
    assert!(!score_matches_distribution(1.006, &thirds));
    assert!(!score_matches_distribution(0.994, &thirds));
    // Precision finer than thousandths keeps the exact check.
    assert!(!score_matches_distribution(1.0001, &thirds));
    // Finer bins get no envelope: only the exact recomputed expectation passes.
    assert!(score_matches_distribution(1.001, &[0.333, 0.333, 0.334]));
    assert!(!score_matches_distribution(1.002, &[0.333, 0.333, 0.334]));
}

#[test]
fn a_tenths_score_is_judged_at_the_precision_it_was_reported() {
    // Production 2026-10-17 (jev-latest via typesafe, 16 refusals): bins
    // whose latent expectations span [3.85, 3.89] beside a score the
    // provider rounded to ONE decimal. 3.9 stands for [3.85, 3.95].
    let bins = [0.0, 0.01, 0.01, 0.08, 0.9];
    let (admitted, provider) = score_payload(&bins, 3.9);
    public_decisions(provider, &admitted).expect("a tenths-rounded score of 3.87");
    assert!(score_matches_distribution(3.9, &bins));
    assert!(score_matches_distribution(3.8, &bins));
    // A genuine mismatch at tenths is still refused: 3.7 covers at most
    // 3.75, below every expectation the bins allow.
    assert!(!score_matches_distribution(3.7, &bins));
    let (admitted, provider) = score_payload(&bins, 3.7);
    assert!(public_decisions(provider, &admitted)
        .expect_err("3.7 is not 3.87 at any rounding")
        .safe_message
        .starts_with("decision score does not match its probability distribution ("));
    // A whole number reads at tenths, never as a half-unit allowance: 4
    // covers [3.95, 4.05], above the envelope.
    assert!(!score_matches_distribution(4.0, &bins));
    // Hundredths keep their own half-hundredth allowance.
    assert!(score_matches_distribution(3.89, &bins));
    assert!(!score_matches_distribution(3.91, &bins));
}

#[test]
fn a_rejected_score_names_its_numbers_in_the_ledger_detail() {
    let (admitted, provider) = score_payload(&[0.0, 0.87, 0.13, 0.0, 0.0], 1.5);
    let failure = public_decisions(provider, &admitted).expect_err("inconsistent score");
    assert!(failure
        .safe_message
        .starts_with("decision score does not match its probability distribution ("));
    for fact in [
        "score=1.5",
        "expected=1.13",
        "probabilities=[0,0.87,0.13,0,0]",
        "rounding_envelope=[1.12,1.15]",
        "residual_envelope=[1.12,1.16]",
        "tolerance=0.000001",
    ] {
        assert!(
            failure.safe_message.contains(fact),
            "{fact} in {}",
            failure.safe_message
        );
    }
    let boundary = failure.boundary();
    assert_eq!(
        boundary.safe_message,
        "provider returned a malformed response; retry the request"
    );
    assert!(boundary.provider_detail.unwrap().contains("score=1.5"));
}

#[test]
fn a_score_beside_a_folded_rounding_residual_is_admitted() {
    // Production 2026-10-23 (jev-latest via typesafe, 82 refusals in three
    // hours, numbers only): TypeSafe rounds every bin to hundredths, folds the
    // residual into the highest bin so the total reads exactly one, and keeps
    // the unrounded expectation as the score. Four levels miss the independent
    // envelope by exactly 0.03, five levels by 0.04.
    for (probabilities, score) in [
        (vec![0.49, 0.37, 0.11, 0.03], 0.71),
        (vec![0.62, 0.26, 0.08, 0.04], 0.57),
        (vec![0.73, 0.18, 0.06, 0.03], 0.36),
        (vec![0.47, 0.36, 0.13, 0.04], 0.71),
        (vec![0.44, 0.3, 0.16, 0.1], 0.95),
        (vec![0.6900000000000001, 0.21, 0.06, 0.04], 0.48),
        (vec![0.6799999999999999, 0.2, 0.08, 0.04], 0.45),
        (vec![0.3, 0.5, 0.15, 0.03, 0.02], 0.93),
        (vec![0.02, 0.01, 0.05, 0.52, 0.4], 3.23),
    ] {
        let independent = score_envelope(&probabilities).expect("unit hundredth bins");
        assert!(
            !score_meets_envelope(score, independent.0, independent.1),
            "{probabilities:?} {score} is outside the independent envelope"
        );
        let (admitted, provider) = score_payload(&probabilities, score);
        let (public, _) = public_decisions(provider.clone(), &admitted)
            .expect("a folded rounding residual explains the score");
        assert_eq!(public["answers"], provider["answers"]);
    }
    // One hundredth further is refused: the folded bin sits at most half a
    // hundredth per other bin from its latent value.
    for (probabilities, score) in [
        (vec![0.49, 0.37, 0.11, 0.03], 0.72),
        (vec![0.49, 0.37, 0.11, 0.03], 0.64),
        (vec![0.3, 0.5, 0.15, 0.03, 0.02], 0.92),
        (vec![0.02, 0.01, 0.05, 0.52, 0.4], 3.22),
    ] {
        let (admitted, provider) = score_payload(&probabilities, score);
        assert!(public_decisions(provider, &admitted)
            .expect_err("beyond the folded residual")
            .safe_message
            .contains("residual_envelope=["));
    }
    // A middle highest bin folds less: [0.3, 0.6, 0.1] spans only [0.79, 0.81]
    // under folding, so 0.82 is still refused.
    assert!(!score_matches_distribution(0.82, &[0.3, 0.6, 0.1]));
    // Bins that do not total exactly one were not folded.
    assert_eq!(residual_envelope(&[0.33, 0.33, 0.33]), None);
    assert_eq!(residual_envelope(&[0.333, 0.333, 0.334]), None);
}

#[test]
fn the_folded_residual_envelope_matches_an_independent_enumeration() {
    // Latent bins on a quarter-hundredth grid (twice the half-hundredth rounding
    // ties), rounded to hundredths with ties either way, then the residual folded
    // into each bin that rounded highest: the extremes of every latent
    // expectation that publishes exactly these bins, in half-hundredths. Shares
    // nothing with the envelope's bound-filling algorithm.
    fn enumerate(probabilities: &[f64]) -> (i64, i64) {
        let published: Vec<i64> = probabilities
            .iter()
            .map(|p| (p * 100.0).round() as i64)
            .collect();
        let count = published.len();
        let mut extremes: Option<(i64, i64)> = None;
        let mut latent = vec![0_i64; count];
        fn walk(
            index: usize,
            latent: &mut Vec<i64>,
            published: &[i64],
            extremes: &mut Option<(i64, i64)>,
        ) {
            let count = published.len();
            if index + 1 == count {
                let last = 400 - latent[..index].iter().sum::<i64>();
                if !(0..=400).contains(&last) {
                    return;
                }
                latent[index] = last;
                let options: Vec<Vec<i64>> = latent
                    .iter()
                    .map(|units| match units % 4 {
                        2 => vec![units / 4, units / 4 + 1],
                        remainder if remainder > 2 => vec![units / 4 + 1],
                        _ => vec![units / 4],
                    })
                    .collect();
                let mut rounded = vec![0_i64; count];
                round_each(0, &options, &mut rounded, latent, published, extremes);
                return;
            }
            let center = published[index] * 4;
            for units in (center - 12).max(0)..=(center + 12).min(400) {
                latent[index] = units;
                walk(index + 1, latent, published, extremes);
            }
        }
        fn round_each(
            index: usize,
            options: &[Vec<i64>],
            rounded: &mut Vec<i64>,
            latent: &[i64],
            published: &[i64],
            extremes: &mut Option<(i64, i64)>,
        ) {
            if index == options.len() {
                let highest = *rounded.iter().max().unwrap();
                let residual = 100 - rounded.iter().sum::<i64>();
                for folded in 0..rounded.len() {
                    if rounded[folded] != highest {
                        continue;
                    }
                    let matches = rounded.iter().enumerate().all(|(i, value)| {
                        published[i] == value + if i == folded { residual } else { 0 }
                    });
                    if matches {
                        // Quarter-hundredths to half-hundredths: exact, the
                        // envelope's extremes sit on the half-hundredth grid.
                        let value: i64 = latent.iter().enumerate().map(|(i, q)| i as i64 * q).sum();
                        *extremes = Some(extremes.map_or((value, value), |(low, high)| {
                            (low.min(value), high.max(value))
                        }));
                    }
                }
                return;
            }
            for choice in &options[index] {
                rounded[index] = *choice;
                round_each(index + 1, options, rounded, latent, published, extremes);
            }
        }
        walk(0, &mut latent, &published, &mut extremes);
        let (low, high) = extremes.expect("the published bins are reachable");
        (low / 2, high / 2)
    }
    for probabilities in [
        vec![0.49, 0.37, 0.11, 0.03],
        vec![0.3, 0.5, 0.15, 0.03, 0.02],
        vec![0.02, 0.01, 0.05, 0.52, 0.4],
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.87, 0.13, 0.0, 0.0],
        vec![0.33, 0.34, 0.01, 0.32],
        vec![0.5, 0.5],
        vec![0.3, 0.6, 0.1],
    ] {
        assert_eq!(
            residual_envelope(&probabilities),
            Some(enumerate(&probabilities)),
            "{probabilities:?}"
        );
    }
}
