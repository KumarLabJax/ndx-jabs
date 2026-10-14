from hdmf.utils import docval, get_docval
from pynwb import register_class
from pynwb.file import Subject


@register_class("JaxStudySubject", "ndx-jabs")
class JaxStudySubject(Subject):
    """Subject with the housing and test-apparatus identifiers recorded for animals in JAX studies."""

    __nwbfields__ = ("housing_id", "test_chamber")

    @docval(
        *get_docval(Subject.__init__),
        {
            "name": "housing_id",
            "type": str,
            "doc": "Identifier of the housing (for example the cage) the animal was kept in.",
            "default": None,
        },
        {
            "name": "test_chamber",
            "type": str,
            "doc": "Identifier of the test chamber (apparatus) the animal was tested in.",
            "default": None,
        },
    )
    def __init__(self, **kwargs):
        housing_id = kwargs.pop("housing_id")
        test_chamber = kwargs.pop("test_chamber")
        super().__init__(**kwargs)
        self.housing_id = housing_id
        self.test_chamber = test_chamber
