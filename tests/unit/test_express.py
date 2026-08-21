"""Black-box tests: the expression layer (pipe + the documented recipe)."""


class TestPipe:
    def test_pipe_threads_left_to_right(self):
        from fpl.express import pipe

        assert pipe(3, lambda x: x + 1, lambda x: x * 2) == 8

    def test_pipe_with_no_steps_returns_input(self):
        from fpl.express import pipe

        assert pipe(7) == 7


def test_express_docstrings_execute():
    """The Data -> [Player] -> [Squad] -> [Score] recipe actually runs, so the
    documented way of writing programs with the domain never drifts."""
    import doctest

    import fpl.express

    results = doctest.testmod(fpl.express)
    assert results.failed == 0, results