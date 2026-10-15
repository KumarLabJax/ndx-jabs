from hdmf.utils import docval, get_docval
from pynwb import register_class

from .jax_study_subject import JaxStudySubject


@register_class("JaxAgingSubject", "ndx-jabs")
class JaxAgingSubject(JaxStudySubject):
    """Subject with the frailty measurements and diet recorded for animals in the JAX aging studies."""

    __nwbfields__ = ("cfi", "cfi_norm", "pll", "diet")

    @docval(
        *get_docval(JaxStudySubject.__init__),
        {
            "name": "cfi",
            "type": ("int", "float"),
            "doc": "Cumulative frailty index score: manually scored item scores (0, 0.5 or 1) summed over 28 items.",
            "default": None,
        },
        {
            "name": "cfi_norm",
            "type": ("int", "float"),
            "doc": "Cumulative frailty index divided by the number of items (28), from 0 to 1.",
            "default": None,
        },
        {
            "name": "pll",
            "type": ("int", "float"),
            "doc": "Proportion of life lived: age at the open-field test divided by age at death. Unitless.",
            "default": None,
        },
        {
            "name": "diet",
            "type": str,
            "doc": "Diet intervention group: AL, 1D, 2D, 20 or 40.",
            "default": None,
        },
    )
    def __init__(self, **kwargs):
        cfi = kwargs.pop("cfi")
        cfi_norm = kwargs.pop("cfi_norm")
        pll = kwargs.pop("pll")
        diet = kwargs.pop("diet")
        super().__init__(**kwargs)
        self.cfi = None if cfi is None else float(cfi)
        self.cfi_norm = None if cfi_norm is None else float(cfi_norm)
        self.pll = None if pll is None else float(pll)
        self.diet = diet
