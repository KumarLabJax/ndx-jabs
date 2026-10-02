# ndx-jabs Extension for NWB

NWB extension for storing JABS (JAX Animal Behavior System) algorithm internals, such as instance segmentation contours

General-purpose JABS outputs belong in tool-agnostic extensions: pose estimates in
[ndx-pose](https://github.com/rly/ndx-pose) and behavior predictions in an ethogram extension.
This extension holds the pieces that are internal to the
[JABS](https://github.com/KumarLabJax/JABS-behavior-classifier) algorithms, so JABS users can keep them in their NWB
files without adding them to the core schema.

![JABS video frame, pose keypoints, and segmentation contour for one mouse](https://raw.githubusercontent.com/KumarLabJax/ndx-jabs/main/docs/images/contour-rendering-example.gif)

*Left: video frame with the region around one mouse boxed. Middle: JABS pose keypoints (12 named points) for that mouse.
Right: the segmentation contour JABS produces for the same mouse, stored by `ContourSeries`.*

## Installation

```bash
pip install ndx-jabs
```

## Neurodata types

### `ContourSeries`

Extends `TimeSeries`. Per-frame instance segmentation contours for one animal, stored as JABS produces them.

| Field | Type | Shape | Description |
|---|---|---|---|
| `data` | `int16` | `(num_times, num_contours, num_vertices, 2)` | Vertex `(x, y)` pixel coordinates in the video frame |
| `external_flag` (optional) | `bool` | `(num_times, num_contours)` | `True` for an external (outer) contour, `False` for an internal contour (hole) or an unused slot |
| `padding_value` (attribute, optional) | `int16` | scalar | Value marking unused contour slots and vertices in `data`; defaults to `-1` |

A segmentation can have several disjoint blobs and holes, so each frame holds up to `num_contours` closed polygons.
Contour and vertex capacities are per-file maxima chosen by the producer, and unused entries are filled with
`padding_value`.

## Usage

```python
import datetime

import numpy as np
from pynwb import NWBHDF5IO, NWBFile

from ndx_jabs import ContourSeries

nwbfile = NWBFile(
    session_description="JABS contours",
    identifier="example",
    session_start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
)

contours = ContourSeries(
    name="contours_mouse0",
    description="JABS segmentation contours for mouse0",
    data=np.full((3600, 4, 319, 2), -1, dtype=np.int16),
    external_flag=np.zeros((3600, 4), dtype=bool),
    unit="pixels",
    rate=30.0,
)
nwbfile.add_acquisition(contours)

with NWBHDF5IO("contours.nwb", "w") as io:
    io.write(nwbfile)
```

The extension spec is cached in files that use it, so readers without `ndx-jabs` installed can still open them.

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                            # install the package and dev tools into .venv
uv run python src/spec/create_extension_spec.py    # regenerate spec/ after editing the spec
uv run pytest                                      # run the tests
uv sync --group test --resolution lowest-direct    # test against the minimum supported dependency versions
uv build                                           # build the wheel and source distribution
```

## Releasing

1. Update `version` in `pyproject.toml` and in `src/spec/create_extension_spec.py`, then run
   `uv run python src/spec/create_extension_spec.py` to regenerate `spec/`.
2. Update `CHANGELOG.md`, commit, and merge to `main`.
3. Tag the commit and push the tag:

   ```bash
   git tag 0.1.0
   git push origin 0.1.0
   ```

The `Publish` workflow checks that the tag, `pyproject.toml` and spec versions match, runs the tests, publishes to
[PyPI](https://pypi.org/project/ndx-jabs/) using trusted publishing, and creates a GitHub release.

---
This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).
