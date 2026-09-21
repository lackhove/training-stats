"""Tests for the local directory source."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from training_stats.errors import SourceError
from training_stats.sources.local import LocalDirectorySource


def make_source(root: Path, **kwargs: object) -> LocalDirectorySource:
    values: dict[str, object] = {"name": "local", "path": root}
    values.update(kwargs)
    return LocalDirectorySource(**values)  # type: ignore[arg-type]


def test_empty_directory(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    assert source.list_files() == []


def test_missing_directory(tmp_path: Path) -> None:
    source = make_source(tmp_path / "missing")
    with pytest.raises(SourceError, match="does not exist"):
        source.list_files()


def test_path_is_file(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")
    with pytest.raises(SourceError, match="not a directory"):
        make_source(target).list_files()


def test_recursive_listing(tmp_path: Path) -> None:
    (tmp_path / "a.fit").write_bytes(b"aa")
    nested = tmp_path / "2026" / "05"
    nested.mkdir(parents=True)
    (nested / "b.fit").write_bytes(b"bb")
    source = make_source(tmp_path)
    paths = [item.path for item in source.list_files()]
    assert sorted(paths) == ["2026/05/b.fit", "a.fit"]


def test_non_recursive_listing(tmp_path: Path) -> None:
    (tmp_path / "a.fit").write_bytes(b"aa")
    nested = tmp_path / "2026"
    nested.mkdir()
    (nested / "b.fit").write_bytes(b"bb")
    source = make_source(tmp_path, recursive=False)
    paths = [item.path for item in source.list_files()]
    assert paths == ["a.fit"]


def test_extension_filtering(tmp_path: Path) -> None:
    (tmp_path / "ride.FIT").write_bytes(b"upper")
    (tmp_path / "ride2.fit").write_bytes(b"lower")
    (tmp_path / "notes.txt").write_bytes(b"nope")
    (tmp_path / "ride.fit.gpx").write_bytes(b"nope")
    source = make_source(tmp_path)
    paths = [item.path for item in source.list_files()]
    assert sorted(paths) == ["ride.FIT", "ride2.fit"]


def test_duplicate_names_in_subdirectories(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "same.fit").write_bytes(b"1")
    (tmp_path / "b" / "same.fit").write_bytes(b"2")
    source = make_source(tmp_path)
    paths = sorted(item.path for item in source.list_files())
    assert paths == ["a/same.fit", "b/same.fit"]


def test_metadata_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "a.fit"
    target.write_bytes(b"abc")
    source = make_source(tmp_path)
    first = source.list_files()
    second = source.list_files()
    assert first == second
    assert first[0].size == 3


def test_read_file_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "a.fit").write_bytes(b"payload")
    source = make_source(tmp_path)
    files = source.list_files()
    assert source.read_file(files[0]) == b"payload"


def test_read_file_rejects_escape(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    from training_stats.models import SourceFile

    evil = SourceFile(source_name="local", path="../outside.fit")
    with pytest.raises(SourceError, match="outside the configured directory"):
        source.read_file(evil)


def test_read_unreadable_file(tmp_path: Path) -> None:
    (tmp_path / "a.fit").write_bytes(b"data")
    os.chmod(tmp_path / "a.fit", 0o000)
    source = make_source(tmp_path)
    try:
        files = source.list_files()
        with pytest.raises(SourceError, match="cannot read"):
            source.read_file(files[0])
    finally:
        os.chmod(tmp_path / "a.fit", 0o644)


def test_symlinks_skipped_by_default(tmp_path: Path) -> None:
    (tmp_path / "real.fit").write_bytes(b"real")
    other = tmp_path / "other"
    other.mkdir()
    (other / "link.fit").symlink_to(tmp_path / "real.fit")
    source = make_source(tmp_path)
    paths = [item.path for item in source.list_files()]
    assert paths == ["real.fit"]
