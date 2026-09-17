# Plan preprocessingu Brain MRI

## `src/brain_mri/data/preprocessing.py`

- [x] `clip_percentiles()`
- [x] Bezpieczna obsługa pustego wolumenu w `z_score_normalise()`
- [x] `crop_or_pad_slice()`
- [x] `validate_slice()`
- [x] `cut_volume()`
- [x] `get_slice()`

## `src/brain_mri/data/nifti_loader.py`

- [x] `load_patient()`
- [x] Clipping intensywności przed normalizacją w `load_patient()`
- [x] Normalizacja obrazów MRI
- [x] Zachowanie maski bez normalizacji i clippingu
- [x] Sprawdzanie zgodności rozmiaru obrazu i maski

## `src/brain_mri/data/slice_extraction.py`

- [ ] Utworzenie pliku
- [ ] `process_single_slice()`
- [ ] Pobieranie slice’a obrazu i maski
- [ ] Crop lub padding obrazu i maski
- [ ] Walidacja slice’a
- [ ] `get_valid_slice_indices()`
- [ ] Wykorzystanie zakresów z `patients_z_boundaries.json`
- [ ] `extract_patient_slices()`

## `configs/data_config.yaml`

- [ ] `target_size: [224, 224]`
- [ ] Zakres `clip_percentiles: [1.0, 99.0]`
- [ ] Lista modalności: `T1`, `T1CE`, `T2`, `FLAIR`
- [ ] Ścieżka do `patients_z_boundaries.json`

## `data/patients_z_boundaries.json`

- [x] Utworzenie pliku
- [ ] Wykorzystanie zakresów osi Z w preprocessingu

## `src/brain_mri/data/split.py`

- [x] Podział danych na poziomie pacjentów
- [x] Podział na train, validation i test
- [x] Zapis podziału do pliku JSON
- [x] Wykonanie podziału przed generowaniem slice’ów

## Pipeline

- [x] Wczytanie pacjenta
- [x] `clip_percentiles()`
- [x] `z_score_normalise()`
- [x] `cut_volume()`
- [x] `get_slice()`
- [x] `crop_or_pad_slice()`
- [ ] `validate_slice()`
- [ ] Zwrot lub zapis gotowych slice’ów

## Później

- [ ] Augmentacje dla zbioru treningowego
- [ ] Dataset PyTorch
- [ ] Konwersja do HDF5
- [ ] Testy jednostkowe
