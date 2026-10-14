# ndx-jabs Extension for NWB

NWB extension for storing JABS (JAX Animal Behavior System) algorithm internals, such as instance segmentation contours, and subject metadata for JAX studies

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
| `data` | numeric (JABS writes `int16`) | `(num_frames, num_contours, num_vertices, 2)` | Vertex `(x, y)` positions. Only the first `vertex_count` vertices of each slot are real; the rest is padding with no meaning |
| `vertex_count` | `uint32` | `(num_frames, num_contours)` | Number of valid vertices in each contour slot; `0` means the slot is empty on that frame |
| `is_external` | `bool` | `(num_frames, num_contours)` | `True` for an external (outer) boundary, `False` for an internal boundary (a hole) |
| `contour_group` (optional) | `uint32` | `(num_frames, num_contours)` | Groups contours into connected components within a frame, so each hole stays with the part that contains it |
| `reference_frame` | text | scalar | What `(0, 0)` is and which way each axis increases |

A segmentation can have several disjoint blobs and holes, so each frame holds up to `num_contours` closed polygons.
Contour and vertex capacities are per-file maxima chosen by the producer. `vertex_count` is authoritative, so a
consumer never needs to look for a padding value. `unit` defaults to `pixels`. Name each series after the instance it
describes, and store it alongside the `PoseEstimation` for that instance.

### `JaxStudySubject`

Extends `Subject`. Adds identifiers recorded for animals in JAX studies. Every added field is optional and is omitted
when the value is unknown or not applicable.

| Field | Type | Description |
|---|---|---|
| `housing_id` | text | Identifier of the housing (for example the cage) the animal was kept in |
| `test_chamber` | text | Identifier of the test chamber (apparatus) the animal was tested in |

### `JaxAgingSubject`

Extends `JaxStudySubject`. Adds the frailty measurements and experimental covariates recorded for animals in the JAX
aging studies, on top of `housing_id` and `test_chamber`. Every added field is optional.

| Field | Type | Description |
|---|---|---|
| `cfi` | `float64` | Cumulative frailty index (CFI) score: a trained technician scores each of 28 items 0 (normal), 0.5 (partly affected) or 1 (affected) and the scores are summed. Not adjusted for tester, batch or diet |
| `cfi_norm` | `float64` | `cfi` divided by the number of items (28), from 0 to 1; higher is frailer |
| `pll` | `float64` | Proportion of life lived: age at the open-field test divided by age at death, from 0 to 1. Omitted for animals not followed to death |
| `diet` | text | Diet intervention group: `AL` (ad libitum), `1D` or `2D` (fasting one day or two consecutive days per week), `20` or `40` (caloric restriction at 20% or 40% of baseline ad libitum food intake) |
| `batch` | text | Data-collection batch of the frailty assessment (`Batch1` to `Batch3` for B6J, `Old` or `New` for DO) |
| `tester` | text | Anonymized ID (`Scorer1` to `Scorer7`) of the trained expert who assigned the manual frailty-index score after the 1-hour open-field session. IDs are only comparable within one numbering: the combined-cohort and single-cohort analyses number the same scorers differently |

Values are stored as given. The ranges above and the relationship between `cfi` and `cfi_norm` are documented but not
validated, and neither is the set of `diet` values.

Pass either as the `subject` of an `NWBFile`:

```python
from ndx_jabs import JaxAgingSubject

subject = JaxAgingSubject(
    subject_id="AgedB6-0396",
    species="Mus musculus",
    sex="M",
    age="P55W",
    cfi=7.5,
    cfi_norm=0.267857143,
    diet="AL",
    batch="Batch1",
    tester="Scorer3",
    housing_id="cage-42",
)
```

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

num_frames, num_contours, num_vertices = 3600, 4, 319
contours = ContourSeries(
    name="contours_mouse0",
    description="JABS segmentation contours for mouse0",
    data=np.full((num_frames, num_contours, num_vertices, 2), -1, dtype=np.int16),
    vertex_count=np.zeros((num_frames, num_contours), dtype=np.uint32),
    is_external=np.zeros((num_frames, num_contours), dtype=bool),
    reference_frame="(0, 0) is the top left corner of the video frame, x increases rightward and y downward",
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
