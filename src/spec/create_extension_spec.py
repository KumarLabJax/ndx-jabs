# -*- coding: utf-8 -*-
from pathlib import Path

from pynwb.spec import NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBNamespaceBuilder, export_spec


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        name="""ndx-jabs""",
        version="""0.1.0""",
        doc=(
            "NWB extension for storing JABS (JAX Animal Behavior System) algorithm internals, "
            "such as instance segmentation contours"
        ),
        author=[
            "The Jackson Laboratory",
        ],
        contact=[
            "glen.beane@jax.org",
        ],
    )
    ns_builder.include_namespace("core")
    
    contour_series = NWBGroupSpec(
        neurodata_type_def="ContourSeries",
        neurodata_type_inc="TimeSeries",
        doc=(
            "Per-frame instance segmentation contours for a single animal, as produced by JABS. Each frame holds "
            "up to num_contours closed polygons, because a segmentation can have several disjoint blobs and holes. "
            "Contours and vertex lists are padded to fixed capacities with the padding value; the capacities are the "
            "per-file maxima chosen by the producer."
        ),
        attributes=[
            NWBAttributeSpec(
                name="padding_value",
                doc="Value used in `data` to mark unused contour slots and unused vertices within a contour.",
                dtype="int16",
                default_value=-1,
                required=False,
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name="data",
                doc=(
                    "Contour vertices as (x, y) pixel coordinates in the video frame. Unused contour slots and "
                    "vertices are filled with `padding_value`."
                ),
                dtype="int16",
                shape=[[None, None, None, 2]],
                dims=[["num_times", "num_contours", "num_vertices", "x, y"]],
            ),
            NWBDatasetSpec(
                name="external_flag",
                doc=(
                    "True where the contour is an external (outer) boundary, False where it is an internal "
                    "boundary (a hole) or the slot is unused."
                ),
                dtype="bool",
                shape=[[None, None]],
                dims=[["num_times", "num_contours"]],
                quantity="?",
            ),
        ],
    )

    new_data_types = [contour_series]

    # export the spec to yaml files in the root spec folder
    output_dir = str((Path(__file__).parent.parent.parent / "spec").absolute())
    export_spec(ns_builder, new_data_types, output_dir)


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()
