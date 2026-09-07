from fractions import Fraction

from so.experiments.e000102_symbolic_pod_algebra_reduction import (
    GenericDependencyProduct,
    candidate_lift,
    provenance_baseline,
    run,
    specialized_lifecycle_product_update,
)


def test_candidate_matches_generic_provenance_coefficients():
    coeffs = [Fraction(2, 3), Fraction(3, 5), Fraction(5, 7), Fraction(7, 11)]
    for family in ("additive", "pairwise", "full_product"):
        assert candidate_lift(family, coeffs) == provenance_baseline(family, coeffs)


def test_full_product_expansion_has_registered_growth():
    coeffs = [Fraction(i + 2, i + 3) for i in range(8)]
    poly = candidate_lift("full_product", coeffs)
    assert len(poly) == 256
    for edited in range(8):
        assert sum(1 for monomial in poly if monomial[edited]) == 128


def test_compact_specialized_update_matches_generic_dependency_work():
    leaves = [Fraction(3, 2), Fraction(4, 3), Fraction(5, 4), Fraction(6, 5), Fraction(7, 6)]
    edited = 2
    new_value = Fraction(11, 7)
    candidate_output, candidate_ops = specialized_lifecycle_product_update(
        leaves, edited, new_value
    )
    generic = GenericDependencyProduct(leaves)
    generic_ops = generic.update(edited, new_value)
    assert candidate_output == generic.output
    assert candidate_ops == generic_ops


def test_registered_small_run_passes():
    summary = run(seed_count=2, sizes=[4, 6])
    assert summary["kill_screen_pass"] is True
    assert summary["exact_mismatches"] == 0
    assert summary["representation_mismatches"] == 0
    assert summary["compact_output_mismatches"] == 0
    assert summary["compact_work_mismatches"] == 0
