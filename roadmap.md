# Roadmap — System detekcji i klasyfikacji guzów mózgu (BraTS 2024)

> Zaznaczaj wykonane zadania wpisując `x` w nawiasie: `- [x]`.
> W VS Code checkboxy renderują się wizualnie w podglądzie Markdown (Ctrl+Shift+V / Cmd+Shift+V).

---

## Etap 0 — Środowisko i przygotowanie (0.5 tyg.)

- [X] Repozytorium Git + struktura folderów projektu
- [X] Środowisko: nibabel, h5py, monai/SimpleITK, torch/tensorflow, scikit-learn, matplotlib
- [X] Konfiguracja black, flake8/ruff, isort
- [X] Pobranie i weryfikacja integralności zbiorów BraTS-GLI i BraTS-MEN
- [X] Montowanie Google Drive (Drive for Desktop / rclone) + symlink w `data/raw/`
- [X] Plik `.env` z `DATA_ROOT` + `.env.example` w repo

## Etap 1 — Eksploracyjna analiza danych (EDA) (1–1.5 tyg.)

- [X] Wczytanie próbek NIfTI, sprawdzenie modalności (T1, T1CE, T2, FLAIR)
- [X] Analiza rozdzielczości, voxel spacing, orientacji
- [X] Rozkład klas: liczba przypadków GLI vs MEN
- [X] Wizualizacja slice'ów MRI + overlay masek segmentacyjnych
- [X] Sprawdzenie braków danych, artefaktów, duplikatów pacjentów
- [] Zapisanie wykresów EDA (histogramy, przykładowe obrazy) do `results/figures`

## Etap 2 — Preprocessing i etykietowanie (1 tydz.)

- [ ] Decyzja: klasyfikacja 2D (slice'y) czy 3D (cały wolumen) — z uzasadnieniem
- [ ] Normalizacja intensywności (z-score / percentile clipping)
- [ ] Podział train/val/test stratyfikowany per pacjent
- [ ] Zapis podziału do `data_splits.json`

## Etap 3 — Moduł konwersji do HDF5 (1–1.5 tyg.)

- [ ] Implementacja `h5_converter.py` (iteracja po pacjentach, batch processing)
- [ ] Kompresja (gzip) i dobór chunkingu
- [ ] Osobne datasety w H5: `images`, `labels`, `patient_ids`
- [ ] Test jednostkowy round-trip (dane po odczycie = oryginał)
- [ ] Benchmark czasu odczytu: NIfTI vs H5
- [ ] Wykres/tabela porównawcza rozmiaru i czasu odczytu do pracy

## Etap 4 — Budowa modelu CNN (1.5 tyg.)

- [ ] Implementacja baseline (prosty custom CNN)
- [ ] Implementacja modelu docelowego (np. ResNet 2D/3D)
- [ ] `model_factory.py` — prosta funkcja `get_model(config)`
- [ ] Ważona funkcja straty (class weighting / focal loss)
- [ ] Augmentacje danych (rotacje, flipy, ew. elastic deformation)

## Etap 5 — Trening (1.5 tyg.)

- [ ] Pętla treningowa z early stopping i checkpointingiem
- [ ] Logowanie loss/accuracy (TensorBoard lub CSV)
- [ ] Ręczny tuning 2–3 wariantów hiperparametrów
- [ ] Zapis najlepszego modelu do `models_checkpoints/`

## Etap 6 — Ewaluacja (1 tydz.)

- [ ] Implementacja metryk: accuracy, sensitivity, specificity
- [ ] Macierz pomyłek (confusion matrix)
- [ ] Test na nietkniętym wcześniej zbiorze testowym
- [ ] Zapis wyników jako tabela CSV do `results/tables`

## Etap 7 — Wizualizacje do pracy dyplomowej (0.5–1 tydz.)

- [ ] Krzywe uczenia (train/val loss i accuracy)
- [ ] Confusion matrix jako heatmapa
- [ ] Grad-CAM / saliency maps (opcjonalnie, 3–4 przykłady)
- [ ] Diagram architektury systemu (data flow + model pipeline)
- [ ] Finalna tabela zbiorcza metryk gotowa do wklejenia w pracy

## Etap 8 — Dokumentacja i porządki (0.5 tydz.)

- [ ] README z instrukcją odtworzenia eksperymentu
- [ ] Docstringi w spójnym stylu w całym kodzie
- [ ] Sprzątanie configów i notebooków
- [ ] Ostateczny przegląd zgodności kodu z PEP8

---

**Postęp:** 0 / 43 zadań