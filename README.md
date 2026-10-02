# ndx-jabs

An [NWB](https://www.nwb.org/) extension for storing the internal data products of
[JABS](https://github.com/KumarLabJax/JABS-behavior-classifier) (JAX Animal Behavior System)
that have no tool-agnostic representation in NWB.

General-purpose JABS outputs belong in tool-agnostic extensions: pose estimates in
[ndx-pose](https://github.com/rly/ndx-pose) and behavior predictions in an ethogram
extension. This extension holds the pieces that are internal to the JABS algorithms, so JABS
users can keep them in their NWB files without adding them to the core schema.

## Neurodata types

### `ContourSeries`

Extends `TimeSeries`. Per-frame instance segmentation contours for one animal, stored as JABS
produces them.

| Field | Type | Shape | Description |
|---|---|---|---|
| `data` | `int16` | `(num_times, num_contours, num_vertices, 2)` | Vertex `(x, y)` pixel coordinates in the video frame |
| `external_flag` (optional) | `bool` | `(num_times, num_contours)` | `True` for an external (outer) contour, `False` for an internal contour (hole) or an unused slot |
| `padding_value` (attribute, optional) | `int16` | scalar | Value marking unused contour slots and vertices in `data`; defaults to `-1` |

A segmentation can have several disjoint blobs and holes, so each frame holds up to
`num_contours` closed polygons. Contour and vertex capacities are per-file maxima chosen by
the producer, and unused entries are filled with `padding_value`.

## Usage

```python
import numpy as np
from pynwb import NWBHDF5IO
from ndx_jabs import ContourSeries

contours = ContourSeries(
    name="contours_mouse0",
    description="JABS segmentation contours for mouse0",
    data=np.full((3600, 4, 319, 2), -1, dtype=np.int16),
    external_flag=np.zeros((3600, 4), dtype=bool),
    unit="pixels",
    rate=30.0,
)
nwbfile.add_acquisition(contours)
```

The extension spec is cached in files that use it, so readers without `ndx-jabs` installed
can still open them.

## Development

```bash
uv sync
uv run python src/spec/create_extension_spec.py   # regenerate spec/ after editing the spec
uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
