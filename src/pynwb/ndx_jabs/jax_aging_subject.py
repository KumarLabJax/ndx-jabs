from hdmf.utils import docval, get_docval
from pynwb import register_class

from .jax_study_subject import JaxStudySubject


@register_class("JaxAgingSubject", "ndx-jabs")
class JaxAgingSubject(JaxStudySubject):
    """Subject with the frailty measurements and covariates recorded for animals in the JAX aging studies."""

    __nwbfields__ = ("cfi", "cfi_norm", "pll", "diet", "batch", "tester")

    @docval(
        *get_docval(JaxStudySubject.__init__),
        {
            "name": "cfi",
            "type": float,
            "doc": "Cumulative frailty index score: manually scored item scores (0, 0.5 or 1) summed over 28 items.",
            "default": None,
        },
        {
            "name": "cfi_norm",
            "type": float,
            "doc": "Cumulative frailty index divided by the number of items (28), from 0 to 1.",
            "default": None,
        },
        {
            "name": "pll",
            "type": float,
            "doc": "Proportion of life lived: age at the open-field test divided by age at death. Unitless.",
            "default": None,
        },
        {
            "name": "diet",
            "type": str,
            "doc": "Diet intervention group: AL, 1D, 2D, 20 or 40.",
            "default": None,
        },
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
        cfi = kwargs.pop("cfi")
        cfi_norm = kwargs.pop("cfi_norm")
        pll = kwargs.pop("pll")
        diet = kwargs.pop("diet")
        batch = kwargs.pop("batch")
        tester = kwargs.pop("tester")
        super().__init__(**kwargs)
        self.cfi = cfi
        self.cfi_norm = cfi_norm
        self.pll = pll
        self.diet = diet
        self.batch = batch
        self.tester = tester
