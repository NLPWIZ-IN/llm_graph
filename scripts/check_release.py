"""Validate a release candidate against package metadata and PyPI history."""

import json
import os
from pathlib import Path
import re
import tomllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from packaging.version import Version


def check_version(candidate: str, tag: str, releases: dict) -> None:
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", candidate):
        raise ValueError("Release version must use the MAJOR.MINOR.PATCH format.")
    if tag != f"v{candidate}":
        raise ValueError(f"Release tag {tag!r} must match v{candidate}.")
    published = [Version(value) for value, files in releases.items() if files]
    if published and Version(candidate) <= max(published):
        raise ValueError(
            f"Candidate {candidate} must be newer than the highest published "
            f"PyPI version, {max(published)} (including yanked releases)."
        )


def fetch_releases(project: str) -> dict:
    url = f"https://pypi.org/pypi/{quote(project, safe='')}/json"
    try:
        with urlopen(url, timeout=30) as response:
            data = json.load(response)
    except HTTPError as error:
        with error:
            if error.code != 404:
                raise
            print(f"PyPI has no project named {project!r}; checking an initial release.")
            return {}
    releases = data["releases"]
    if not isinstance(releases, dict) or any(
        not isinstance(files, list) for files in releases.values()
    ):
        raise ValueError("PyPI returned an invalid releases mapping.")
    return releases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    candidate = project["version"]
    tag = os.environ["RELEASE_TAG"]
    check_version(candidate, tag, {})
    check_version(candidate, tag, fetch_releases(project["name"]))
    print(f"Release check passed for {project['name']} {candidate}.")


if __name__ == "__main__":
    main()
