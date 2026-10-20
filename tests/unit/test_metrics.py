import pytest

from verisearch.evaluation.metrics import binary_metrics, bootstrap_accuracy


def test_known_metrics() -> None:
    result = binary_metrics([False, True], [0.1, 0.9])
    assert result["accuracy_at_0.5"] == 1
    assert result["auroc"] == 1 and result["auprc"] == 1
    assert result["brier"] == pytest.approx(0.01)
    assert result["ece"] == pytest.approx(0.1)


def test_perfect_and_undefined() -> None:
    result = binary_metrics([True, True], [1.0, 1.0])
    assert result["auroc"] is None and result["auprc"] is None
    assert result["ece"] == 0 and result["brier"] == 0
    with pytest.raises(ValueError):
        binary_metrics([], [])
    with pytest.raises(ValueError):
        binary_metrics([True], [float("nan")])


def test_problem_bootstrap() -> None:
    outcomes = [[True, False], [True, True], [False, False]]
    assert bootstrap_accuracy(outcomes, 1000, 42) == bootstrap_accuracy(outcomes, 1000, 42)
    assert bootstrap_accuracy([[True], [True]]) == (1, 1)
    low, high = bootstrap_accuracy(outcomes)
    assert 0 <= low <= 0.5 <= high <= 1


def test_tied_ranking_scores_do_not_depend_on_order() -> None:
    from verisearch.evaluation.metrics import ranking_metrics

    labels = [False, True, False, True]
    scores = [0.1, 0.4, 0.4, 0.9]
    auc, ap = ranking_metrics(labels, scores)
    assert auc == pytest.approx(0.875)
    assert ap == pytest.approx(5 / 6)
    assert ranking_metrics(labels[::-1], scores[::-1]) == (auc, ap)
    assert ranking_metrics([False, True], [0.5, 0.5]) == (0.5, 0.5)


def test_paired_difference_retains_problem_and_seed_pairs() -> None:
    from verisearch.evaluation.metrics import bootstrap_difference

    assert bootstrap_difference([[(True, True)], [(False, False)]]) == (0, 0, 0)
    assert bootstrap_difference([[(True, False)], [(True, False)]]) == (1, 1, 1)
    pairs = [[(True, False), (False, True)], [(True, True), (True, False)]]
    estimate, low, high = bootstrap_difference(pairs, 2000, 42)
    assert estimate == pytest.approx(0.25)
    assert low <= estimate <= high
    assert bootstrap_difference(pairs, 2000, 42) == (estimate, low, high)
    reverse = [[(b, a) for a, b in row] for row in pairs]
    reverse_mean, reverse_low, reverse_high = bootstrap_difference(reverse, 2000, 42)
    assert reverse_mean == -estimate
    assert reverse_low == -high and reverse_high == -low
    with pytest.raises(ValueError):
        bootstrap_difference([[]])


def test_answer_presence_shortcut_is_reported_separately() -> None:
    from verisearch.evaluation.metrics import answer_presence_diagnostics

    diagnostic = answer_presence_diagnostics(
        [False, True, False], [0.01, 0.95, 0.9], [None, "2", "3"]
    )
    assert diagnostic["answer_missing"] == 1
    assert diagnostic["present_incorrect"] == 1
    assert diagnostic["presence_heuristic_metrics"]["accuracy_at_0.5"] == pytest.approx(2 / 3)
    assert diagnostic["verifier_metrics_on_present_answers"]["count"] == 2
    only_correct_present = answer_presence_diagnostics([False, True], [0.1, 0.9], [None, "2"])
    assert only_correct_present["verifier_metrics_on_present_answers"]["auroc"] is None
    all_missing = answer_presence_diagnostics([False], [0.1], [None])
    assert all_missing["verifier_metrics_on_present_answers"] is None
    with pytest.raises(ValueError):
        answer_presence_diagnostics([True], [0.9], [])
