from importlib.resources import files
from pynwb import load_namespaces

# Get path to the namespace.yaml file with the expected location when installed not in editable mode
__location_of_this_file = files(__name__)
__spec_path = __location_of_this_file / "spec" / "ndx-jabs.namespace.yaml"

# If that path does not exist, we are likely running in editable mode. Use the local path instead
if not __spec_path.exists():
    __spec_path = __location_of_this_file.parent.parent.parent / "spec" / "ndx-jabs.namespace.yaml"

# Load the namespace
load_namespaces(str(__spec_path))

# Import the classes only after the namespace is loaded: they register themselves with PyNWB on import.
from .contour_series import ContourSeries  # noqa: E402

__all__ = [
    "ContourSeries",
]

# Remove these functions/modules from the package
del load_namespaces, files, __location_of_this_file, __spec_path
