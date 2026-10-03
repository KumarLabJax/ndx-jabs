from hdmf.utils import docval, get_docval
from pynwb import register_class
from pynwb.base import TimeSeries


@register_class("ContourSeries", "ndx-jabs")
class ContourSeries(TimeSeries):
    """Polygon contours outlining a segmented instance over time, as produced by JABS."""

    __nwbfields__ = ("vertex_count", "is_external", "contour_group", "reference_frame")

    @docval(
        *get_docval(TimeSeries.__init__, "name"),
        {
            "name": "data",
            "type": ("array_data", "data"),
            "shape": (None, None, None, 2),
            "doc": (
                "Contour vertex positions (x, y), shape (num_frames, num_contours, num_vertices, 2). Only the first "
                "'vertex_count' vertices of each contour slot hold a position; the rest is padding."
            ),
        },
        {
            "name": "vertex_count",
            "type": ("array_data", "data"),
            "shape": (None, None),
            "doc": "Number of valid vertices in each contour slot, shape (num_frames, num_contours).",
        },
        {
            "name": "is_external",
            "type": ("array_data", "data"),
            "shape": (None, None),
            "doc": "True where a contour slot is an external boundary, False where it is a hole.",
        },
        {
            "name": "reference_frame",
            "type": str,
            "doc": "What (0, 0) of the vertex coordinates is and which way each axis increases.",
        },
        {
            "name": "contour_group",
            "type": ("array_data", "data"),
            "shape": (None, None),
            "doc": "Index grouping contours into connected components within a frame.",
            "default": None,
        },
        {
            "name": "unit",
            "type": str,
            "doc": "Base unit of measurement for the vertex positions.",
            "default": "pixels",
        },
        *[arg for arg in get_docval(TimeSeries.__init__) if arg["name"] not in ("name", "data", "unit")],
    )
    def __init__(self, **kwargs):
        vertex_count = kwargs.pop("vertex_count")
        is_external = kwargs.pop("is_external")
        reference_frame = kwargs.pop("reference_frame")
        contour_group = kwargs.pop("contour_group")
        super().__init__(**kwargs)
        self.vertex_count = vertex_count
        self.is_external = is_external
        self.reference_frame = reference_frame
        self.contour_group = contour_group
