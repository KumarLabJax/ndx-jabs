"""NWB extension for JABS (JAX Animal Behavior System) algorithm internals."""

import os
from importlib.resources import files

from pynwb import get_class, load_namespaces

# Spec location when installed normally (the wheel ships spec/ inside the package).
__location_of_this_file = files(__name__)
__spec_path = __location_of_this_file / "spec" / "ndx-jabs.namespace.yaml"

# In an editable install the spec lives at the repository root instead.
if not os.path.exists(__spec_path):
    __spec_path = __location_of_this_file.parent.parent.parent / "spec" / "ndx-jabs.namespace.yaml"

load_namespaces(str(__spec_path))

ContourSeries = get_class("ContourSeries", "ndx-jabs")

__all__ = ["ContourSeries"]
