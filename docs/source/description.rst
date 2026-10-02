Overview
========

``ndx-jabs`` stores the internal data products of JABS (JAX Animal Behavior System) that have no tool-agnostic
representation in NWB. General-purpose JABS outputs, such as pose estimates and behavior predictions, belong in
tool-agnostic extensions.

``ContourSeries`` holds per-frame instance segmentation contours for one animal, stored as JABS produces them.
