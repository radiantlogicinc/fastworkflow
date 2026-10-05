"""The gate that stops a browser-free run from reporting browser coverage.

Nothing here is mocked: discovery runs against this suite's real DOM tests,
the prerequisite check runs against the real environment, and the outcome
reader parses a JUnit report written by a real pytest run. The one thing these
tests deliberately never contain is the literal name of the jsdom environment
variable — the gate discovers browser checks by looking for it in the source,
so spelling it here would enrol this file in its own validation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests import browser_validation
from tests.browser_validation import (
    BrowserPrerequisiteMissing,
    JSDOM_ROOT_VAR,
    ValidationReport,
    check_prerequisites,
    discover_required_dom_tests,
    outcome_for,
    read_junit_outcomes,
)


def _fake_dom_test_file(directory: Path, body: str) -> Path:
    """A test file shaped like this suite's DOM tests, outside the repo."""
    path = directory / "test_fake_browser_check.py"
    path.write_text(
        "import os\n"
        "import pytest\n"
        "\n"
        "\n"
        "def _drive_the_dom():\n"
        f"    return os.environ.get({JSDOM_ROOT_VAR!r})\n"
        "\n"
        "\n"
        f"{body}",
        encoding="utf-8",
    )
    return path


def test_the_real_dom_tests_are_all_discovered():
    required = discover_required_dom_tests()

    assert len(required) >= 17
    assert "tests/test_chatbot_hierarchy.py::test_hierarchy_dom_clicks" in required
    assert all("::test_" in node for node in required)
    for node in required:
        source = Path(node.split("::")[0]).read_text(encoding="utf-8")
        assert JSDOM_ROOT_VAR in source


def test_a_browser_check_reached_through_a_helper_is_still_discovered(tmp_path):
    """Two real DOM tests call the harness through a module-local helper.

    Discovery that only looked for the variable in the test body would report
    coverage of a smaller suite than exists, which is the failure this whole
    module is about.
    """
    path = _fake_dom_test_file(
        tmp_path,
        "def test_through_a_helper():\n    assert _drive_the_dom() is not None\n",
    )

    required = discover_required_dom_tests([path])

    assert required == [f"{path}::test_through_a_helper"]


def test_the_helper_itself_is_not_reported_as_a_check(tmp_path):
    path = _fake_dom_test_file(
        tmp_path,
        "def test_through_a_helper():\n    assert _drive_the_dom() is not None\n",
    )

    assert not any(
        "_drive_the_dom" in node for node in discover_required_dom_tests([path])
    )


class TestChecksWrittenInsideAClass:
    """A DOM test written as a method has to be discovered like any other.

    It used not to be: discovery walked a module's top-level functions only,
    so a class-nested browser check skipped itself on a machine without jsdom
    and the gate reported nothing missing -- the exact outcome this gate
    exists to make impossible (fix-17mu). These tests are themselves methods,
    which is the cheapest way to keep that discovery honest.
    """

    def test_a_method_is_discovered_with_its_class_in_the_node_id(self, tmp_path):
        path = _fake_dom_test_file(
            tmp_path,
            "class TestInTheBrowser:\n"
            "    def test_in_a_class(self):\n"
            "        assert _drive_the_dom() is not None\n",
        )

        assert discover_required_dom_tests([path]) == [
            f"{path}::TestInTheBrowser::test_in_a_class"
        ]

    def test_a_sibling_helper_on_the_class_is_followed(self, tmp_path):
        """Two hops: the method calls a method that calls a module helper."""
        path = _fake_dom_test_file(
            tmp_path,
            "class TestInTheBrowser:\n"
            "    def _open_the_page(self):\n"
            "        return _drive_the_dom()\n"
            "\n"
            "    def test_through_a_sibling(self):\n"
            "        assert self._open_the_page() is not None\n",
        )

        assert discover_required_dom_tests([path]) == [
            f"{path}::TestInTheBrowser::test_through_a_sibling"
        ]

    def test_a_helper_reached_through_the_class_name_is_followed(self, tmp_path):
        path = _fake_dom_test_file(
            tmp_path,
            "class TestInTheBrowser:\n"
            "    @staticmethod\n"
            "    def _open_the_page():\n"
            "        import os\n"
            f"        return os.environ.get({JSDOM_ROOT_VAR!r})\n"
            "\n"
            "    def test_through_the_class(self):\n"
            "        assert TestInTheBrowser._open_the_page() is not None\n",
        )

        assert discover_required_dom_tests([path]) == [
            f"{path}::TestInTheBrowser::test_through_the_class"
        ]

    def test_a_method_that_drives_no_dom_is_not_enrolled(self, tmp_path):
        """Over-discovery is not the safe direction.

        A test required here that never needed jsdom would be free to skip for
        its own reasons, and the gate would fail a run that was fine.
        """
        path = _fake_dom_test_file(
            tmp_path,
            "class TestInTheBrowser:\n"
            "    def test_in_a_class(self):\n"
            "        assert _drive_the_dom() is not None\n"
            "\n"
            "    def test_plain(self):\n"
            "        assert True\n"
            "\n"
            "    def test_on_another_object(self):\n"
            "        assert str(self).strip() is not None\n",
        )

        assert discover_required_dom_tests([path]) == [
            f"{path}::TestInTheBrowser::test_in_a_class"
        ]

    def test_a_nested_class_is_addressed_through_both_names(self, tmp_path):
        path = _fake_dom_test_file(
            tmp_path,
            "class TestOuter:\n"
            "    class TestInner:\n"
            "        def test_deep(self):\n"
            "            assert _drive_the_dom() is not None\n",
        )

        assert discover_required_dom_tests([path]) == [
            f"{path}::TestOuter::TestInner::test_deep"
        ]

    @pytest.mark.parametrize(
        "body,why",
        [
            (
                "class NotCollected:\n"
                "    def test_looks_like_one(self):\n"
                "        assert _drive_the_dom() is not None\n",
                "pytest collects Test* classes only",
            ),
            (
                "class TestWithInit:\n"
                "    def __init__(self):\n"
                "        self.page = None\n"
                "\n"
                "    def test_never_runs(self):\n"
                "        assert _drive_the_dom() is not None\n",
                "pytest refuses to collect a class with __init__",
            ),
        ],
    )
    def test_a_class_pytest_would_not_collect_is_not_required(
        self, tmp_path, body, why
    ):
        """Requiring an uncollectable node id would fail the gate forever.

        It would be reported as never having run, which is true and useless:
        there is no check there to run.
        """
        path = _fake_dom_test_file(tmp_path, body)

        assert discover_required_dom_tests([path]) == [], why

    def test_the_class_helper_itself_is_not_reported_as_a_check(self, tmp_path):
        path = _fake_dom_test_file(
            tmp_path,
            "class TestInTheBrowser:\n"
            "    def _open_the_page(self):\n"
            "        return _drive_the_dom()\n"
            "\n"
            "    def test_through_a_sibling(self):\n"
            "        assert self._open_the_page() is not None\n",
        )

        assert not any(
            "_open_the_page" in node for node in discover_required_dom_tests([path])
        )


class TestOutcomesOfChecksInsideAClass:
    """Discovery is only half of it: the outcome has to be found again.

    A JUnit report has one dotted field for the module and the class, so a
    class-nested case whose name is cut in the wrong place reads as a node id
    naming a file that does not exist -- "never reported", which fails the
    gate without saying why. Every outcome below comes from a real pytest run.
    """

    @staticmethod
    def _junit_for(path: Path, directory: Path) -> Path:
        junit = directory / "report.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(path),
                "-p",
                "no:randomly",
                "-q",
                f"--junitxml={junit}",
            ],
            cwd=directory,
            capture_output=True,
            check=False,
        )
        return junit

    @pytest.fixture
    def outcomes(self, tmp_path):
        path = _fake_dom_test_file(
            tmp_path,
            "class TestReported:\n"
            "    def test_that_passes(self):\n"
            "        assert _drive_the_dom() is None or True\n"
            "\n"
            "    def test_that_skips(self):\n"
            "        pytest.skip('the dependency is not really here')\n"
            "\n"
            "    @pytest.mark.parametrize('case', ['ran', 'did not'])\n"
            "    def test_with_a_case_that_skips(self, case):\n"
            "        if case == 'did not':\n"
            "            pytest.skip('one case never ran')\n"
            "\n"
            "    @pytest.mark.parametrize('case', ['ran', 'broke'])\n"
            "    def test_with_a_case_that_fails(self, case):\n"
            "        assert case == 'ran'\n",
        )
        return path, read_junit_outcomes(self._junit_for(path, tmp_path))

    def test_a_method_is_found_under_its_own_node_id(self, outcomes):
        path, reported = outcomes

        assert (
            outcome_for(f"{path}::TestReported::test_that_passes", reported)
            == "passed"
        )

    def test_the_class_is_not_folded_into_the_file_path(self, outcomes):
        _path, reported = outcomes

        assert all(
            "TestReported.py" not in key and "TestReported/" not in key
            for key in reported
        ), reported

    def test_a_method_that_skipped_reads_as_skipped(self, outcomes):
        path, reported = outcomes

        assert (
            outcome_for(f"{path}::TestReported::test_that_skips", reported)
            == "skipped"
        )

    def test_one_case_skipping_makes_the_whole_check_skipped(self, outcomes):
        """Half a browser check having run is not the check having run."""
        path, reported = outcomes

        assert (
            outcome_for(f"{path}::TestReported::test_with_a_case_that_skips", reported)
            == "skipped"
        )

    def test_one_case_failing_makes_the_whole_check_failed(self, outcomes):
        path, reported = outcomes

        assert (
            outcome_for(f"{path}::TestReported::test_with_a_case_that_fails", reported)
            == "failed"
        )

    def test_a_check_that_was_never_written_has_no_outcome(self, outcomes):
        path, reported = outcomes

        assert outcome_for(f"{path}::TestReported::test_absent", reported) is None


def test_the_gate_fails_when_a_check_inside_a_class_skips(tmp_path):
    """End to end, with a real pytest run underneath.

    This is the whole point of the follow-up: before class-nested discovery,
    this run reported success having executed no browser code.
    """
    try:
        check_prerequisites()
    except BrowserPrerequisiteMissing as missing:
        pytest.skip(str(missing))

    path = _fake_dom_test_file(
        tmp_path,
        "class TestInTheBrowser:\n"
        "    def test_that_skips(self):\n"
        "        _drive_the_dom()\n"
        "        pytest.skip('the dependency is not really here')\n",
    )

    report = browser_validation.validate([path])

    assert report.required == [f"{path}::TestInTheBrowser::test_that_skips"]
    assert report.skipped == report.required
    assert report.missing == [], "the outcome was found, and it was a skip"
    assert report.ok is False


def test_an_unconfigured_jsdom_location_is_an_error_not_a_skip(monkeypatch):
    monkeypatch.delenv(JSDOM_ROOT_VAR, raising=False)

    with pytest.raises(BrowserPrerequisiteMissing) as failure:
        check_prerequisites()

    assert JSDOM_ROOT_VAR in str(failure.value)


def test_a_configured_location_without_jsdom_is_reported_as_such(tmp_path):
    with pytest.raises(BrowserPrerequisiteMissing) as failure:
        check_prerequisites(str(tmp_path))

    assert "node_modules/jsdom" in str(failure.value)


def test_a_run_whose_checks_all_skipped_is_not_a_successful_run():
    report = ValidationReport(
        required=["tests/test_x.py::test_dom"],
        skipped=["tests/test_x.py::test_dom"],
        pytest_exit_code=0,
    )

    assert report.ok is False
    assert "skipped" in report.summary()


def test_discovering_no_checks_at_all_is_also_a_failure():
    report = ValidationReport(required=[], pytest_exit_code=0)

    assert report.ok is False
    assert "no browser checks were discovered" in report.summary()


def test_outcomes_are_read_back_from_a_real_pytest_report(tmp_path):
    """A skipped browser check reads as skipped, not as absent."""
    path = _fake_dom_test_file(
        tmp_path,
        "def test_that_passes():\n"
        "    assert _drive_the_dom() is None or True\n"
        "\n"
        "\n"
        "def test_that_skips():\n"
        "    _drive_the_dom()\n"
        "    pytest.skip('the dependency is not really here')\n",
    )
    junit = tmp_path / "report.xml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(path),
            "-p",
            "no:randomly",
            "-q",
            f"--junitxml={junit}",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )

    outcomes = read_junit_outcomes(junit)

    assert outcome_for(f"{path}::test_that_passes", outcomes) == "passed"
    assert outcome_for(f"{path}::test_that_skips", outcomes) == "skipped"
    assert outcome_for(f"{path}::test_never_written", outcomes) is None


def test_the_gate_reports_failure_when_a_required_check_skips(tmp_path):
    """End to end, with a real pytest run underneath.

    The prerequisites are the real ones: without a configured jsdom this test
    has nothing to say and says so, which is the skip the ordinary suite is
    allowed to keep.
    """
    try:
        check_prerequisites()
    except BrowserPrerequisiteMissing as missing:
        pytest.skip(str(missing))

    path = _fake_dom_test_file(
        tmp_path,
        "def test_that_skips():\n"
        "    _drive_the_dom()\n"
        "    pytest.skip('the dependency is not really here')\n",
    )

    report = browser_validation.validate([path])

    assert report.ok is False
    assert report.skipped == [f"{path}::test_that_skips"]
    assert report.passed == []


def test_the_gate_does_not_count_itself_as_a_browser_check():
    required = discover_required_dom_tests()

    assert not any("test_browser_validation_gate" in node for node in required)
