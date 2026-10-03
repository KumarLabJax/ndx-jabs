"""Tests for the ContourSeries neurodata type."""

import datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, validate

from ndx_jabs import ContourSeries

REFERENCE_FRAME = "(0, 0) is the top left corner of the video frame, x increases rightward and y downward"


def _nwbfile() -> NWBFile:
    return NWBFile(
        session_description="test",
        identifier="test-id",
        session_start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


def _series(n_frames: int = 6, n_contours: int = 3, n_vertices: int = 8, **overrides) -> ContourSeries:
    """Build a ContourSeries with one real five-vertex external contour per frame and -1 padding elsewhere."""
    data = np.full((n_frames, n_contours, n_vertices, 2), -1, dtype=np.int16)
    data[:, 0, :5] = np.arange(10, dtype=np.int16).reshape(5, 2)
    vertex_count = np.zeros((n_frames, n_contours), dtype=np.uint32)
    vertex_count[:, 0] = 5
    is_external = np.zeros((n_frames, n_contours), dtype=bool)
    is_external[:, 0] = True
    kwargs = {
        "name": "contours",
        "data": data,
        "vertex_count": vertex_count,
        "is_external": is_external,
        "reference_frame": REFERENCE_FRAME,
        "rate": 30.0,
    }
    kwargs.update(overrides)
    return ContourSeries(**kwargs)


def test_constructor_stores_fields() -> None:
    series = _series()
    assert series.data.shape == (6, 3, 8, 2)
    assert series.vertex_count.shape == (6, 3)
    assert series.is_external.shape == (6, 3)
    assert series.reference_frame == REFERENCE_FRAME


def test_unit_defaults_to_pixels() -> None:
    assert _series().unit == "pixels"


def test_contour_group_is_optional() -> None:
    assert _series().contour_group is None


@pytest.mark.parametrize("missing", ["vertex_count", "is_external", "reference_frame"])
def test_required_fields(missing: str) -> None:
    kwargs = {
        "name": "contours",
        "data": np.zeros((2, 1, 4, 2), dtype=np.int16),
        "vertex_count": np.zeros((2, 1), dtype=np.uint32),
        "is_external": np.zeros((2, 1), dtype=bool),
        "reference_frame": REFERENCE_FRAME,
        "rate": 30.0,
    }
    del kwargs[missing]
    with pytest.raises(TypeError):
        ContourSeries(**kwargs)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32], ids=["int16", "int32", "float32"])
def test_round_trip_and_validate(tmp_path: Path, dtype: type) -> None:
    series = _series()
    expected = series.data.astype(dtype)
    series = _series(data=expected, contour_group=np.zeros((6, 3), dtype=np.uint32))
    nwbfile = _nwbfile()
    nwbfile.add_acquisition(series)
    path = tmp_path / "contours.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)

    with NWBHDF5IO(path, "r") as io:
        assert validate(io=io) == []
        read = io.read().acquisition["contours"]
        assert isinstance(read, ContourSeries)
        assert read.data.dtype == dtype
        np.testing.assert_array_equal(read.data[:], expected)
        np.testing.assert_array_equal(read.vertex_count[:], series.vertex_count)
        np.testing.assert_array_equal(read.is_external[:], series.is_external)
        np.testing.assert_array_equal(read.contour_group[:], np.zeros((6, 3), dtype=np.uint32))
        assert read.reference_frame == REFERENCE_FRAME


def test_wrong_dimensionality_is_rejected() -> None:
    with pytest.raises(ValueError, match="data"):
        _series(data=np.zeros((4, 3), dtype=np.int16))


def test_wrong_vertex_count_dimensionality_is_rejected() -> None:
    with pytest.raises(ValueError, match="vertex_count"):
        _series(vertex_count=np.zeros(6, dtype=np.uint32))
