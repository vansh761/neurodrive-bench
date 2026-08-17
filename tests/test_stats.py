from neurodrive_bench.metrics.stats import aggregate, mean, stdev


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0


def test_mean_basic():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_stdev_of_single_value_is_zero():
    assert stdev([5.0]) == 0.0


def test_stdev_of_empty_is_zero():
    assert stdev([]) == 0.0


def test_stdev_of_identical_values_is_zero():
    assert stdev([3.0, 3.0, 3.0]) == 0.0


def test_stdev_matches_known_value():
    # population stdev of [2, 4, 4, 4, 5, 5, 7, 9] is 2.0
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert abs(stdev(values) - 2.0) < 1e-9


def test_aggregate_contains_expected_keys():
    result = aggregate([1.0, 2.0, 3.0])
    assert set(result.keys()) == {"mean", "std", "min", "max", "n"}
    assert result["mean"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0
    assert result["n"] == 3.0


def test_aggregate_of_empty_list_does_not_raise():
    result = aggregate([])
    assert result["mean"] == 0.0
    assert result["n"] == 0.0
