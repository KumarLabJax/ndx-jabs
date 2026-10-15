"""Tests for the JaxAgingStudyMetaData neurodata type."""

import datetime
from pathlib import Path

import pytest
from pynwb import NWBHDF5IO, NWBFile, validate
from pynwb.file import LabMetaData

from ndx_jabs import JaxAgingStudyMetaData, JaxAgingSubject

FIELDS = {"batch": "Batch1", "tester": "Scorer3"}


def _nwbfile() -> NWBFile:
    return NWBFile(
        session_description="test",
        identifier="test-id",
        session_start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


def _round_trip(tmp_path: Path, metadata: JaxAgingStudyMetaData) -> JaxAgingStudyMetaData:
    nwbfile = _nwbfile()
    nwbfile.add_lab_meta_data(metadata)
    path = tmp_path / "metadata.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(path, "r") as io:
        assert validate(io=io) == []
        read = io.read().lab_meta_data[metadata.name]
        assert isinstance(read, JaxAgingStudyMetaData)
        return read


def test_is_lab_meta_data() -> None:
    assert isinstance(JaxAgingStudyMetaData(), LabMetaData)


def test_default_name() -> None:
    assert JaxAgingStudyMetaData().name == "jax_aging"


def test_constructor_stores_fields() -> None:
    metadata = JaxAgingStudyMetaData(**FIELDS)
    for name, value in FIELDS.items():
        assert getattr(metadata, name) == value


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_fields_are_optional(name: str) -> None:
    assert getattr(JaxAgingStudyMetaData(), name) is None


def test_round_trip_all_fields(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, JaxAgingStudyMetaData(**FIELDS))
    assert read.name == "jax_aging"
    for name, value in FIELDS.items():
        assert getattr(read, name) == value


def test_round_trip_omits_absent_fields(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, JaxAgingStudyMetaData(batch="Old"))
    assert read.batch == "Old"
    assert read.tester is None


def test_round_trip_with_no_fields(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, JaxAgingStudyMetaData())
    assert read.batch is None
    assert read.tester is None


def test_values_are_stored_as_given(tmp_path: Path) -> None:
    read = _round_trip(tmp_path, JaxAgingStudyMetaData(batch="anything", tester="Scorer99"))
    assert (read.batch, read.tester) == ("anything", "Scorer99")


def test_wrong_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="tester"):
        JaxAgingStudyMetaData(tester=3)


def test_coexists_with_jax_aging_subject(tmp_path: Path) -> None:
    """One file carries a JaxAgingSubject and the session-level metadata together and validates."""
    nwbfile = NWBFile(
        session_description="test",
        identifier="test-id",
        session_start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        subject=JaxAgingSubject(subject_id="M1", species="Mus musculus", sex="M", age="P55W", cfi=7.5, diet="AL"),
    )
    nwbfile.add_lab_meta_data(JaxAgingStudyMetaData(**FIELDS))
    path = tmp_path / "both.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(path, "r") as io:
        assert validate(io=io) == []
        read = io.read()
        assert isinstance(read.subject, JaxAgingSubject)
        assert read.subject.cfi == pytest.approx(7.5)
        assert read.lab_meta_data["jax_aging"].tester == "Scorer3"
