# Final Project – Birds Biodiversity Temporal Trends

## Overview

This capstone project asks you to investigate the long-term monitoring data collected in the Birds Biodiversity programme. Your objective is to quantify how biodiversity indicators have evolved, provide sound statistical uncertainty assessments, and highlight species-specific stories that emerge from the dataset.

The brief outlines **what** must be addressed without prescribing **how** you should proceed. Choose statistical methods, visualisations, and modelling strategies that you consider appropriate, and justify every major decision.

You will work with the STOC monitoring extract provided at `projects/birds-biodiversity/data/raw/Observations 2012-2025.xlsx`. Build your own helper code or reuse any open-source statistical libraries you deem suitable.

## Core Deliverables

1. **Technical report** (PDF or Markdown) summarising your approach, indicators studied, and key findings.
2. **Reproducible analysis code** (notebook(s) and/or scripts).
3. **Figures and tables** supporting your conclusions.

## Key Questions to Address

### 1. Dataset Familiarisation and Descriptive Analysis

- Describe the dataset structure (dimensions, key columns, time span, identifiers such as transects/observers/species).
- Produce descriptive statistics showing how the main variables are distributed. Pay particular attention to observer effort because it influences the interpretation of all subsequent results.
- Discuss data quality considerations (missing values, outliers, anomalous periods, etc.).

### 2. Multi-Year Indicator Trends

- Choose at least **three** biodiversity or sampling indicators you consider informative. Explain your selection.
- Compute annual estimates for 2014–2025 and supply confidence intervals (method and level are up to you, but justify the choice).
- Quantify and interpret temporal trends using appropriate models or descriptive tools.

### 3. Species-Level Evolution

- Select a subset of species and examine how their counts/presence evolve over time.
- Provide uncertainty assessments and discuss ecological or operational explanations for the patterns you observe.

### 4. Synthesis and Recommendations

- Summarise insights about the monitoring programme, highlighting converging or diverging evidence across indicators.
- Provide recommendations for future data collection or management where possible. Lack of clear actions will not be penalised, but sloppy analysis will.
- Reflect on limitations (data quality, modelling assumptions, sensitivity analyses).

## Guidance

- The aim is to demonstrate statistical creativity and rigour rather than to discover groundbreaking ecological phenomena.
- Novel indicators or visualisations are welcome if they are well motivated and correctly implemented.
- Incorrect conclusions driven by flawed assumptions or buggy code will be penalised heavily. Validate every step.
- Use standard statistical libraries as needed, but you are accountable for every line of code.
- Keep visualisations polished. If warnings remain, explain why they cannot or should not be suppressed.
- Comment your code thoroughly so reviewers can understand your reasoning without guesswork.

## Suggested Workflow

1. Explore and clean the data.
2. Define indicators and species of interest with justification.
3. Implement analyses (structure notebooks/scripts clearly).
4. Validate results (diagnostics, sensitivity checks, reproducibility runs).
5. Assemble the final report, figures, tables, and README.

## Evaluation Criteria

- **Statistical soundness**
- **Insight and interpretation**
- **Clarity and reproducibility**
- **Creativity and initiative**

## Submission

- Package all deliverables (report, code, figures, tables, README) into a single archive named `FinalProject_GroupName_Lastname1_Lastname2[_Lastname3].zip`.
- Email the archive to `stephane.rivaud@universite-paris-saclay.fr` **before 6 November, 23:59 (Paris time)**. Late submissions are not accepted.
- The README must describe the archive contents and provide reproduction instructions (environment, entry point, runtime notes).
- You may document the study in a PDF report or in notebook form (ensure notebooks are self-contained).
- Each group (2 or 3 students) must list all members and student IDs in both the README and the report/notebook.
- Oral defences will take place between 7 and 14 November (10-minute presentation + 5-minute Q&A).

Good luck—use this project to showcase the statistical toolkit you have acquired!
