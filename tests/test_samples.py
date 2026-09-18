import json
import random

from vllm_bench_eval.results import parse_result_file
from vllm_bench_eval.samples import (
    VLLM_DEFAULT_DATASET_SEED,
    align_requests_to_samples,
    build_sonnet_prompts,
    custom_dataset_order,
    load_samples,
    make_sample_id,
    write_samples,
)


def test_load_samples(samples_file):
    samples = load_samples(samples_file)
    assert [s.prompt for s in samples] == ["prompt-0", "prompt-1", "prompt-2"]
    assert [s.line_index for s in samples] == [0, 1, 2]
    assert [s.sample_id for s in samples] == [
        make_sample_id(i, f"prompt-{i}") for i in range(3)
    ]


def test_custom_dataset_order_matches_vllm_shuffle():
    """vLLM does `random.seed(0); random.shuffle(self.data)` then takes the head."""
    n, k = 8, 5
    data = [{"prompt": f"p{i}"} for i in range(n)]
    random.seed(0)
    random.shuffle(data)
    expected = [int(item["prompt"][1:]) for item in data[:k]]
    assert custom_dataset_order(n, k, seed=0) == expected


def test_custom_dataset_order_replicates_oversampling():
    """When num_requests > dataset size vLLM pads via random.choices()."""
    n, k = 3, 7
    data = list(range(n))
    random.seed(0)
    random.shuffle(data)
    sampled = data[:k]
    random.seed(0)
    sampled = sampled + random.choices(sampled, k=k - len(sampled))
    assert custom_dataset_order(n, k, seed=0) == sampled
    assert len(custom_dataset_order(n, k, seed=0)) == k


def test_custom_dataset_order_is_a_permutation():
    order = custom_dataset_order(8, 8, seed=0)
    assert sorted(order) == list(range(8))


def test_align_requests_to_samples(result_file, samples_file):
    result = parse_result_file(result_file)
    samples = load_samples(samples_file)
    assert align_requests_to_samples(result.requests, samples) is True

    order = custom_dataset_order(len(samples), len(result.requests))
    assert [r.prompt for r in result.requests] == [samples[i].prompt for i in order]
    assert [r.sample_id for r in result.requests] == [samples[i].sample_id for i in order]


def test_align_requests_refuses_non_custom_dataset(result_file, samples_file):
    result = parse_result_file(result_file)
    samples = load_samples(samples_file)
    assert align_requests_to_samples(result.requests, samples, dataset_name="sharegpt") is False
    assert all(r.prompt is None for r in result.requests)


def test_write_and_reload_roundtrip(tmp_path):
    path = write_samples(tmp_path / "out.jsonl", ["a", "b"])
    assert [json.loads(line)["prompt"] for line in path.read_text().splitlines()] == ["a", "b"]
    assert len(load_samples(path)) == 2


def test_build_sonnet_prompts_is_deterministic():
    text = "\n".join(f"line {i}" for i in range(80))
    a = build_sonnet_prompts(text, num_samples=4, lines_per_prompt=8, seed=1234)
    b = build_sonnet_prompts(text, num_samples=4, lines_per_prompt=8, seed=1234)
    assert a == b
    assert len(a) == 4
    assert all(len(p.splitlines()) == 8 for p in a)


# --- review fixes ----------------------------------------------------------


def test_alignment_ignores_benchmark_seed():
    """vLLM shuffles a custom dataset with DEFAULT_SEED=0 whatever --seed says.

    align_requests_to_samples must therefore expose no seed parameter at all.
    """
    import inspect

    assert "seed" not in inspect.signature(align_requests_to_samples).parameters
    # and the two orders really do differ, so passing the wrong seed *would*
    # have silently mislabelled every request
    assert custom_dataset_order(8, 8, seed=0) != custom_dataset_order(8, 8, seed=42)
    assert custom_dataset_order(8, 8, seed=VLLM_DEFAULT_DATASET_SEED) == [4, 1, 5, 2, 0, 3, 7, 6]


def test_alignment_uses_default_seed_even_for_a_seeded_run(result_file, samples_file):
    from vllm_bench_eval.results import parse_result_file

    result = parse_result_file(result_file)
    samples = load_samples(samples_file)
    assert align_requests_to_samples(result.requests, samples) is True
    expected = custom_dataset_order(len(samples), len(result.requests),
                                    seed=VLLM_DEFAULT_DATASET_SEED)
    assert [r.prompt for r in result.requests] == [samples[i].prompt for i in expected]


def test_sample_id_is_content_derived(tmp_path):
    """Same line index + different text must not reuse the same sample_id."""
    a = write_samples(tmp_path / "a.jsonl", ["alpha", "beta"])
    b = write_samples(tmp_path / "b.jsonl", ["ALPHA", "beta"])
    ids_a = [s.sample_id for s in load_samples(a)]
    ids_b = [s.sample_id for s in load_samples(b)]
    assert ids_a[0] != ids_b[0]     # content changed -> id changed
    assert ids_a[1] == ids_b[1]     # same line, same text -> stable id
    assert len(set(ids_a)) == 2


def test_sample_id_disambiguates_duplicate_prompts(tmp_path):
    path = write_samples(tmp_path / "dup.jsonl", ["same", "same"])
    ids = [s.sample_id for s in load_samples(path)]
    assert len(set(ids)) == 2


def test_sample_id_is_stable_across_calls():
    assert make_sample_id(3, "hello") == make_sample_id(3, "hello")
    assert make_sample_id(3, "hello").startswith("s0003-")
