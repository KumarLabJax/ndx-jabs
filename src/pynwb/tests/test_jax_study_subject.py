"""Tests for the JaxStudySubject neurodata type."""

import datetime
from pathlib import Path

import pytest
from pynwb import NWBHDF5IO, NWBFile, validate
from pynwb.file import Subject

from ndx_jabs import JaxStudySubject

STUDY_FIELDS = {"housing_id": "cage-42", "test_chamber": "chamber-A"}


def _nwbfile(subject: Subject) -> NWBFile:
    return NWBFile(
        session_description="test",
        identifier="test-id",
        session_start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        subject=subject,
    )


def _subject(**overrides) -> JaxStudySubject:
    kwargs = {"subject_id": "AgedB6-0396", "species": "Mus musculus", "sex": "M", "age": "P55W"}
    kwargs.update(overrides)
    return JaxStudySubject(**kwargs)


def _round_trip(tmp_path: Path, subject: JaxStudySubject) -> JaxStudySubject:
    path = tmp_path / "subject.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(_nwbfile(subject))
    with NWBHDF5IO(path, "r") as io:
        assert validate(io=io) == []
        read = io.read().subject
        assert isinstance(read, JaxStudySubject)
        return read


def test_is_a_subject() -> None:
    assert isinstance(_subject(), Subject)


def test_constructor_stores_fields() -> None:
    subject = _subject(**STUDY_FIELDS)
    for name, value in STUDY_FIELDS.items():
        assert getattr(subject, name) == value


@pytest.mark.parametrize("name", sorted(STUDY_FIELDS))
def test_study_fields_are_optional(name: str) -> None:
    assert getattr(_subject(), name) is None


def test_round_trip_all_fields(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, _subject(**STUDY_FIELDS))
    assert read.subject_id == "AgedB6-0396"
    assert read.age == "P55W"
    for name, value in STUDY_FIELDS.items():
        assert getattr(read, name) == value


def test_round_trip_with_one_field(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, _subject(housing_id="cage-42"))
    assert read.housing_id == "cage-42"
    assert read.test_chamber is None


def test_round_trip_with_no_study_fields(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, _subject())
    assert all(getattr(read, name) is None for name in STUDY_FIELDS)


def test_wrong_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="housing_id"):
        _subject(housing_id=42)
