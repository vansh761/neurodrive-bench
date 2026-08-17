from neurodrive_bench.data.storage import split_dataset


def make_records(num_episodes: int, steps_per_episode: int) -> list[dict]:
    records = []
    for ep in range(num_episodes):
        for step in range(steps_per_episode):
            records.append({"episode_id": f"ep_{ep}", "step": step, "value": ep * 1000 + step})
    return records


def test_no_episode_appears_in_more_than_one_split():
    records = make_records(num_episodes=20, steps_per_episode=10)
    train, val, test = split_dataset(records, train_ratio=0.7, val_ratio=0.2, seed=1)

    train_eps = {r["episode_id"] for r in train}
    val_eps = {r["episode_id"] for r in val}
    test_eps = {r["episode_id"] for r in test}

    assert train_eps.isdisjoint(val_eps)
    assert train_eps.isdisjoint(test_eps)
    assert val_eps.isdisjoint(test_eps)


def test_all_records_are_preserved_across_splits():
    records = make_records(num_episodes=15, steps_per_episode=8)
    train, val, test = split_dataset(records, train_ratio=0.6, val_ratio=0.2, seed=7)
    assert len(train) + len(val) + len(test) == len(records)


def test_split_is_deterministic_for_a_given_seed():
    records = make_records(num_episodes=20, steps_per_episode=5)
    train1, val1, test1 = split_dataset(records, seed=42)
    train2, val2, test2 = split_dataset(records, seed=42)

    assert [r["episode_id"] for r in train1] == [r["episode_id"] for r in train2]
    assert [r["episode_id"] for r in val1] == [r["episode_id"] for r in val2]
    assert [r["episode_id"] for r in test1] == [r["episode_id"] for r in test2]


def test_different_seeds_can_produce_different_splits():
    records = make_records(num_episodes=20, steps_per_episode=5)
    train_a, _, _ = split_dataset(records, seed=1)
    train_b, _, _ = split_dataset(records, seed=2)

    eps_a = {r["episode_id"] for r in train_a}
    eps_b = {r["episode_id"] for r in train_b}
    assert eps_a != eps_b


def test_ratios_are_approximately_respected_in_episode_count():
    records = make_records(num_episodes=100, steps_per_episode=3)
    train, val, test = split_dataset(records, train_ratio=0.7, val_ratio=0.2, seed=3)

    train_eps = {r["episode_id"] for r in train}
    val_eps = {r["episode_id"] for r in val}
    test_eps = {r["episode_id"] for r in test}

    assert len(train_eps) == 70
    assert len(val_eps) == 20
    assert len(test_eps) == 10
