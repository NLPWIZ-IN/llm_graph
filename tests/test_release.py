import io
import json
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


GUARD = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_release.py")
)
check_version = GUARD["check_version"]
fetch_releases = GUARD["fetch_releases"]


class ReleaseTests(unittest.TestCase):
    def test_first_release(self):
        check_version("0.1.0", "v0.1.0", {})

    def test_newer_numeric_version(self):
        check_version("1.10.0", "v1.10.0", {"1.9.0": [{}]})

    def test_equal_and_older_releases_rejected(self):
        for candidate in ("1.2.0", "1.3.0"):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                check_version(candidate, f"v{candidate}", {"1.3.0": [{}]})

    def test_yanked_versions_still_count(self):
        with self.assertRaises(ValueError):
            check_version("1.2.0", "v1.2.0", {"2.0.0": [{"yanked": True}]})

    def test_empty_release_entries_are_not_publications(self):
        check_version("1.2.0", "v1.2.0", {"2.0.0": []})

    def test_invalid_version_and_tag_rejected(self):
        for candidate, tag in (
            ("1.2", "v1.2"),
            ("1.2.0rc1", "v1.2.0rc1"),
            ("01.2.0", "v01.2.0"),
            ("1.2.0", "v1.3.0"),
        ):
            with self.subTest(candidate=candidate, tag=tag), self.assertRaises(ValueError):
                check_version(candidate, tag, {})

    def test_pypi_response(self):
        payload = {"releases": {"1.2.0": [{"yanked": False}]}}
        with patch.dict(fetch_releases.__globals__, {
            "urlopen": lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())
        }):
            self.assertEqual(fetch_releases("llm-graph"), payload["releases"])

    def test_only_404_allows_initial_publication(self):
        for code in (404, 403, 500):
            def fail(*args, **kwargs):
                raise HTTPError("https://pypi.org", code, "test error", None, None)

            with patch.dict(fetch_releases.__globals__, {"urlopen": fail}):
                if code == 404:
                    self.assertEqual(fetch_releases("llm-graph"), {})
                else:
                    with self.assertRaises(HTTPError):
                        fetch_releases("llm-graph")

    def test_network_and_response_errors_fail_closed(self):
        def fail(*args, **kwargs):
            raise URLError("connection failed")

        with patch.dict(fetch_releases.__globals__, {"urlopen": fail}):
            with self.assertRaises(URLError):
                fetch_releases("llm-graph")
        for payload in (b"invalid json", b'{"releases": []}', b'{"releases":{"1.0.0":null}}'):
            with patch.dict(fetch_releases.__globals__, {
                "urlopen": lambda *args, **kwargs: io.BytesIO(payload)
            }):
                with self.assertRaises(ValueError):
                    fetch_releases("llm-graph")
