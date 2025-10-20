# Birds Biodiversity Dataset Overview

This note helps you quickly understand the contents of `data/raw/Observations 2012-2025.xlsx` before diving into the analysis.

## Source & Scope
- **Program**: Martinique breeding bird monitoring initiative (Applied Statistics final project).
- **Years covered**: 2012–2025 (inclusive) with one row per observation record.
- **Geography**: Island-wide transects; each observation references a monitoring point.

## What each row represents
- A single visit to a monitoring point by an observer on a given date.
- Consolidates the species observed and the number of individuals recorded during that visit.

## Key columns (high level)
- **`observation_id`** – unique identifier for the record.
- **`date` / `year`** – visit timestamp; use `year` for trend analyses.
- **`observer_id`** – anonymous identifier for the person running the transect.
- **`transect_id` / `point_id`** – location metadata.
- **`species_code` / `species_name`** – species observed.
- **`individual_count`** – number of individuals detected.
- **Effort metrics** (e.g., `visit_duration_minutes`, `points_visited`) – useful when normalising counts.

> Columns may include additional attributes (e.g., weather, notes). Inspect the header row after loading to confirm availability.

## Recommended first steps
1. **Load** the data with pandas:
   ```python
   import pandas as pd
   df = pd.read_excel("projects/birds-biodiversity/data/raw/Observations 2012-2025.xlsx")
   ```
2. **Check schema consistency** with `df.info()` and `df.head()`.
3. **Handle missing values** in effort columns or species counts as needed.
4. **Create summary tables** (species richness, observer workload, yearly totals) to orient your analysis.

## Usage expectations
- Use the workbook as the canonical dataset for the project; do not edit it in place—create derived tables/notebooks instead.
- Cite the dataset in your report as "Martinique Breeding Bird Monitoring, 2012–2025".
- If you need to share interim data, copy slices into a separate file to keep the raw extract intact.

Good luck, and remember to tie your EDA back to the project brief! ⚡️
