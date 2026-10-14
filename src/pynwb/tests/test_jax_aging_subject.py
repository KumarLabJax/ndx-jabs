"""Tests for the JaxAgingSubject neurodata type."""

import datetime
from pathlib import Path

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
