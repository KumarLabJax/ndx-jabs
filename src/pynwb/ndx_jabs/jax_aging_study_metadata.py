from hdmf.utils import docval
from pynwb import register_class
from pynwb.file import LabMetaData


@register_class("JaxAgingStudyMetaData", "ndx-jabs")
class JaxAgingStudyMetaData(LabMetaData):
    """Session-level metadata for the JAX aging and frailty studies."""

    __nwbfields__ = ("batch", "tester")

    @docval(
        {"name": "name", "type": str, "doc": "Name of the metadata group.", "default": "jax_aging"},
        {
            "name": "batch",
            "type": str,
            "doc": "Data-collection batch of the frailty assessment.",
            "default": None,
        },
        {
            "name": "tester",
            "type": str,
            "doc": "Anonymized ID of the scorer who assigned the manual frailty-index score.",
            "default": None,
        },
    )
    def __init__(self, **kwargs):
        batch = kwargs.pop("batch")
        tester = kwargs.pop("tester")
        super().__init__(**kwargs)
        self.batch = batch
        self.tester = tester
