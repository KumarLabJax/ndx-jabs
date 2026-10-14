"""Tests for the JaxAgingSubject neurodata type."""

import datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, validate

from ndx_jabs import JaxAgingSubject, JaxStudySubject

AGING_FIELDS = {
    "cfi": 7.5,
    "cfi_norm": 0.267857143,
    "pll": 0.82,
    "diet": "40",
    "batch": "Batch1",
    "tester": "Scorer3",
}
STUDY_FIELDS = {"housing_id": "cage-42", "test_chamber": "chamber-A"}


def _nwbfile(subject: JaxAgingSubject) -> NWBFile:
    return NWBFile(
        session_description="test",
        identifier="test-id",
        session_start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        subject=subject,
    )


def _subject(**overrides) -> JaxAgingSubject:
    kwargs = {"subject_id": "AgedB6-0396", "species": "Mus musculus", "sex": "M", "age": "P55W"}
    kwargs.update(overrides)
    return JaxAgingSubject(**kwargs)


def _round_trip(tmp_path: Path, subject: JaxAgingSubject) -> JaxAgingSubject:
    path = tmp_path / "subject.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(_nwbfile(subject))
    with NWBHDF5IO(path, "r") as io:
        assert validate(io=io) == []
        read = io.read().subject
        assert isinstance(read, JaxAgingSubject)
        return read


def test_is_a_jax_study_subject() -> None:
    assert isinstance(_subject(), JaxStudySubject)


def test_constructor_stores_fields() -> None:
    subject = _subject(**AGING_FIELDS, **STUDY_FIELDS)
    for name, value in {**AGING_FIELDS, **STUDY_FIELDS}.items():
        assert getattr(subject, name) == value


@pytest.mark.parametrize("name", sorted({**AGING_FIELDS, **STUDY_FIELDS}))
def test_fields_are_optional(name: str) -> None:
    assert getattr(_subject(), name) is None


def test_round_trip_all_fields(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, _subject(**AGING_FIELDS, **STUDY_FIELDS))
    assert read.subject_id == "AgedB6-0396"
    assert read.age == "P55W"
    for name, value in {**AGING_FIELDS, **STUDY_FIELDS}.items():
        assert getattr(read, name) == pytest.approx(value)


def test_round_trip_omits_absent_fields(tmp_path: Path) -> None:
    """A B6 animal with no PLL and no tester round-trips with those fields absent."""
    fields = {k: v for k, v in AGING_FIELDS.items() if k not in ("pll", "tester")}
    read = _round_trip(tmp_path, _subject(**fields))
    assert read.pll is None
    assert read.tester is None
    assert read.cfi == pytest.approx(7.5)


def test_round_trip_with_no_study_fields(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, _subject())
    assert all(getattr(read, name) is None for name in {**AGING_FIELDS, **STUDY_FIELDS})


def test_wrong_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="cfi"):
        _subject(cfi="high")


@pytest.mark.parametrize("name", ["cfi", "cfi_norm", "pll"])
@pytest.mark.parametrize("value", [7, np.int64(1), np.float32(0.5)], ids=["int", "np-int", "np-float"])
def test_numeric_fields_accept_ints_and_numpy_numbers(name: str, value: object) -> None:
    assert getattr(_subject(**{name: value}), name) == value


def test_whole_number_values_round_trip_as_float(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, _subject(cfi=7, cfi_norm=1, pll=1))
    assert (read.cfi, read.cfi_norm, read.pll) == (7.0, 1.0, 1.0)
    assert all(isinstance(v, float) for v in (read.cfi, read.cfi_norm, read.pll))


def test_values_are_stored_as_given_without_range_checks(tmp_path: Path) -> None:
    """The documented ranges and cfi/cfi_norm relationship are not enforced."""
    read = _round_trip(tmp_path, _subject(cfi=7.5, cfi_norm=0.9, pll=1.5, diet="other"))
    assert (read.cfi, read.cfi_norm, read.pll, read.diet) == (7.5, 0.9, 1.5, "other")
