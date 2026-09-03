"""The rules in checks.py are the ones most likely to be 'simplified' away.

Each test below corresponds to a real failure observed during the measurement
work that produced this tool, so a future refactor that collapses three outcomes
into two breaks a test rather than a customer's release gate.
"""

from litetune.checks import Check, CheckSet, Outcome, guard


def test_unchecked_is_not_a_failure():
    c = Check.unchecked("model runs", "binary not found on this platform")
    assert c.outcome is Outcome.UNCHECKED
    assert not c.conclusive


def test_set_with_an_unchecked_item_is_unchecked_overall():
    # A caller promised an answer about the model did not get one, even though
    # nothing observed was wrong.
    s = CheckSet("liveness")
    s.add(Check.passed("exit status", "rc=0", 0))
    s.add(Check.unchecked("output parses", "process never started"))
    assert s.outcome is Outcome.UNCHECKED


def test_empty_set_has_established_nothing():
    # Reporting PASSED for an empty set is the exact mistake this module exists
    # to prevent: no observation was made.
    assert CheckSet("liveness").outcome is Outcome.UNCHECKED


def test_failure_wins_over_pass_but_not_over_unchecked():
    s = CheckSet("quality")
    s.add(Check.passed("a", "ok"))
    s.add(Check.failed("b", "0.51 below floor 0.60", 0.51))
    assert s.outcome is Outcome.FAILED
    s.add(Check.unchecked("c", "held-out data absent"))
    assert s.outcome is Outcome.UNCHECKED


def test_guard_turns_an_exception_into_unchecked():
    with guard("model loads") as out:
        raise FileNotFoundError("libvulkan.so.1")
    assert len(out) == 1
    assert out[0].outcome is Outcome.UNCHECKED
    assert "FileNotFoundError" in out[0].detail


def test_guard_leaves_a_real_result_alone():
    with guard("model loads") as out:
        out.append(Check.passed("model loads", "236 tensors", 236))
    assert out[0].outcome is Outcome.PASSED
