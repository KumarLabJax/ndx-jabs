# Changelog for ndx-jabs

## 0.3.0 (unreleased)

- Add `JaxAgingStudyMetaData`, a `LabMetaData` (default name `jax_aging`) with optional `batch` and `tester` fields.
- **Breaking:** `batch` and `tester` move from `JaxAgingSubject` to `JaxAgingStudyMetaData`. No files used them on the subject.

## 0.2.0

- Add `JaxStudySubject`, a `Subject` with optional `housing_id` and `test_chamber` fields.
- Add `JaxAgingSubject`, a `JaxStudySubject` with optional `cfi`, `cfi_norm`, `pll`, `diet`, `batch` and `tester` fields.

## 0.1.0

- Initial `ContourSeries` neurodata type.
