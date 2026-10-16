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
