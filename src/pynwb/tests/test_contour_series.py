"""Tests for the ContourSeries neurodata type."""

import datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, validate

from ndx_jabs import ContourSeries


def _nwbfile() -> NWBFile:
    return NWBFile(
        session_description="test",
        identifier="test-id",
        session_start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


def _contours(n_frames: int = 6, n_contours: int = 3, n_vertices: int = 8) -> np.ndarray:
    data = np.full((n_frames, n_contours, n_vertices, 2), -1, dtype=np.int16)
    data[:, 0, :5] = np.arange(10, dtype=np.int16).reshape(5, 2)
    return data


def test_constructor_stores_fields() -> None:
    data = _contours()
    flags = np.zeros((6, 3), dtype=bool)
    flags[:, 0] = True
    series = ContourSeries(name="contours", data=data, external_flag=flags, unit="pixels", rate=30.0)
    assert series.data.shape == (6, 3, 8, 2)
    assert series.external_flag.shape == (6, 3)


def test_external_flag_is_optional() -> None:
    series = ContourSeries(name="contours", data=_contours(), unit="pixels", rate=30.0)
    assert series.external_flag is None


def test_padding_value_defaults_to_minus_one() -> None:
    series = ContourSeries(name="contours", data=_contours(), unit="pixels", rate=30.0)
    assert series.padding_value == -1


def test_round_trip_and_validate(tmp_path: Path) -> None:
    data = _contours()
    flags = np.zeros((6, 3), dtype=bool)
    flags[:, 0] = True
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(ContourSeries(name="contours", data=data, external_flag=flags, unit="pixels", rate=30.0))
    path = tmp_path / "contours.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)

    with NWBHDF5IO(path, "r") as io:
        assert validate(io=io) == []
        read = io.read().acquisition["contours"]
        assert isinstance(read, ContourSeries)
        np.testing.assert_array_equal(read.data[:], data)
        np.testing.assert_array_equal(read.external_flag[:], flags)
        assert read.data.dtype == np.int16


@pytest.mark.filterwarnings("ignore::hdmf.build.warnings.IncorrectDatasetShapeBuildWarning")
def test_wrong_dimensionality_fails_validation(tmp_path: Path) -> None:
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(ContourSeries(name="bad", data=np.zeros((4, 3), dtype=np.int16), unit="pixels", rate=30.0))
    path = tmp_path / "bad.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)

    with NWBHDF5IO(path, "r") as io:
        assert validate(io=io) != []
