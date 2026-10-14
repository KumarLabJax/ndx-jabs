# -*- coding: utf-8 -*-
from pathlib import Path

from pynwb.spec import NWBAttributeSpec, NWBDatasetSpec, NWBGroupSpec, NWBNamespaceBuilder, export_spec


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        name="""ndx-jabs""",
        version="""0.2.0""",
        doc=(
            "NWB extension for storing JABS (JAX Animal Behavior System) algorithm internals, "
            "such as instance segmentation contours, and subject metadata for JAX studies"
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
            "Polygon contours outlining a segmented instance over time, as produced by JABS. Each frame holds a "
            "fixed number of contour slots, and 'vertex_count' gives the number of valid vertices in each slot, so "
            "trailing slots and trailing vertices are unused padding. More than one contour may be needed to "
            "describe an instance on a frame: an outer boundary plus one or more holes, or a body that an occluder "
            "splits into disjoint parts. Vertex positions are in the frame of reference described by "
            "'reference_frame'. Name the series after the instance it describes and store it alongside the "
            "PoseEstimation for the same instance."
        ),
        datasets=[
            NWBDatasetSpec(
                name="data",
                doc=(
                    "Contour vertex positions (x, y). Only the first 'vertex_count' vertices of each contour slot "
                    "hold a position; the remaining values are padding and carry no meaning."
                ),
                dtype="numeric",
                shape=[[None, None, None, 2]],
                dims=[["num_frames", "num_contours", "num_vertices", "x, y"]],
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        doc=(
                            "Base unit of measurement for working with the data. The default value is 'pixels'. "
                            "Actual stored values are not necessarily stored in these units. To access the data in "
                            "these units, multiply 'data' by 'conversion'."
                        ),
                        dtype="text",
                        default_value="pixels",
                        required=False,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name="vertex_count",
                doc=(
                    "Number of valid vertices in each contour slot. 0 means the slot holds no contour on that "
                    "frame. No count may exceed the number of vertices each slot of 'data' holds."
                ),
                dtype="uint32",
                shape=[[None, None]],
                dims=[["num_frames", "num_contours"]],
            ),
            NWBDatasetSpec(
                name="is_external",
                doc=(
                    "True where the contour slot is an external boundary, i.e. an outer edge of the instance, and "
                    "False where it is an internal boundary, i.e. a hole. Has no meaning where 'vertex_count' is 0. "
                    "A producer that retrieves only outer boundaries sets this True throughout."
                ),
                dtype="bool",
                shape=[[None, None]],
                dims=[["num_frames", "num_contours"]],
            ),
            NWBDatasetSpec(
                name="contour_group",
                doc=(
                    "Index grouping contours into connected components within a frame. Contours describing the "
                    "same component share a value, and a hole carries the value of the component that contains it, "
                    "so an instance that an occluder splits into disjoint parts keeps each hole with the part it "
                    "belongs to. Values are arbitrary labels that need only be distinct within a frame, not indices "
                    "into anything. Has no meaning where 'vertex_count' is 0. Omit this dataset when the component "
                    "structure is not known; a consumer can then attribute each hole to the smallest external "
                    "contour containing it, which is correct even when one component lies inside another's hole."
                ),
                dtype="uint32",
                shape=[[None, None]],
                dims=[["num_frames", "num_contours"]],
                quantity="?",
            ),
            NWBDatasetSpec(
                name="reference_frame",
                doc=(
                    "Description defining what the zero-position (0, 0) of the vertex coordinates is and which way "
                    "each axis increases, e.g. '(0, 0) is the top left corner of the video frame, x increases "
                    "rightward and y increases downward'. Give the same frame of reference as the "
                    "PoseEstimationSeries objects describing the same instance."
                ),
                dtype="text",
            ),
        ],
    )

    jax_study_subject = NWBGroupSpec(
        neurodata_type_def="JaxStudySubject",
        neurodata_type_inc="Subject",
        doc=(
            "Subject with the housing and test-apparatus identifiers recorded for animals in JAX studies. All "
            "added fields are optional and are omitted when the value is unknown or not applicable."
        ),
        datasets=[
            NWBDatasetSpec(
                name="housing_id",
                doc="Identifier of the housing (for example the cage) the animal was kept in.",
                dtype="text",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="test_chamber",
                doc="Identifier of the test chamber (apparatus) the animal was tested in.",
                dtype="text",
                quantity="?",
            ),
        ],
    )

    jax_aging_subject = NWBGroupSpec(
        neurodata_type_def="JaxAgingSubject",
        neurodata_type_inc="JaxStudySubject",
        doc=(
            "Subject with the frailty measurements and experimental covariates recorded for animals in the JAX "
            "aging and frailty studies. All added fields are optional and are omitted when the value is unknown "
            "or not applicable. Values are stored as given: the ranges and the relationship between 'cfi' and "
            "'cfi_norm' described below are not validated."
        ),
        datasets=[
            NWBDatasetSpec(
                name="cfi",
                doc=(
                    "Cumulative frailty index (CFI) score. A trained technician scores each frailty-index item "
                    "0 (normal), 0.5 (partly affected) or 1 (affected), and the item scores are summed. Not "
                    "adjusted for tester, batch or diet. 'cfi_norm' is this value divided by the number of items "
                    "(28)."
                ),
                dtype="float64",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="cfi_norm",
                doc=(
                    "Cumulative frailty index divided by the number of frailty-index items (28), so it runs from 0 "
                    "to 1 with higher values meaning frailer."
                ),
                dtype="float64",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="pll",
                doc=(
                    "Proportion of life lived (PLL): the animal's age at the open-field test divided by its age "
                    "at death. Unitless, from 0 to 1; values closer to 1 mean the animal was nearer the end of "
                    "life. Omitted for animals that were not followed to death."
                ),
                dtype="float64",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="diet",
                doc=(
                    "Diet intervention group: 'AL' for ad libitum feeding; '1D' or '2D' for fasting one day or two "
                    "consecutive days per week; '20' or '40' for caloric restriction at 20% or 40% of baseline "
                    "ad libitum food intake."
                ),
                dtype="text",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="batch",
                doc=(
                    "Data-collection batch of the frailty assessment. Tester, batch and diet are factors whose "
                    "effects are adjusted out of the frailty scores in analysis."
                ),
                dtype="text",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="tester",
                doc=(
                    "Anonymized ID of the trained expert who assigned the manual frailty-index score after the "
                    "animal's 1-hour open-field session, the human reference that machine-vision frailty "
                    "estimates are compared against. IDs are only comparable within one numbering: the "
                    "combined-cohort and single-cohort analyses number the same scorers differently."
                ),
                dtype="text",
                quantity="?",
            ),
        ],
    )

    new_data_types = [contour_series, jax_study_subject, jax_aging_subject]

    # export the spec to yaml files in the root spec folder
    output_dir = str((Path(__file__).parent.parent.parent / "spec").absolute())
    export_spec(ns_builder, new_data_types, output_dir)


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()
