# Agent Prompt: Insert Slides for Quantiles and QQ-Plots

You are a LaTeX/Beamer coding assistant. Modify `main.tex` to insert a sequence of slides introducing **Quantiles** and **QQ-plots**. 
These slides should be clear, pedagogical, and use visuals to highlight intuition. The section is part of descriptive statistics.

---

## Slide 1 — What is a Quantile?
- Title: *Definition of Quantiles*
- Content:
  - Formal definition: The p-quantile is \\( q_p = \inf \{x : F(x) \geq p \} \\), where F is the CDF.
  - Intuition: “the value below which a fraction p of the data lies.”
- Visual instruction:
  - Plot a CDF of a Normal(0,1).
  - Highlight the 25%, 50%, 75% quantiles: horizontal line at p → vertical projection to the curve → x-axis value.
  - Caption: *Quantiles divide the distribution into portions.*

---

## Slide 2 — Why Quantiles?
- Title: *Why are quantiles useful?*
- Content:
  - Quantiles summarize distributional shape (center, spread, tails).
  - Robust to outliers (median, quartiles).
  - Natural way to compare distributions.
- Visual instruction:
  - Show two CDFs (Normal(0,1) vs Uniform(0,1)).
  - Plot quantiles at p=0.25,0.5,0.75 for both and show differences.
  - Caption: *Comparing quantiles = comparing distributions.*

---

## Slide 3 — What is a QQ-Plot?
- Title: *Definition of QQ-Plot*
- Content:
  - QQ-plot = plot quantiles of one distribution against quantiles of another.
  - If distributions are the same → points lie on line y=x.
  - Deviations show differences in location, scale, or tails.
- Visual instruction:
  - Simple diagram: axes labeled "Quantiles of Dist A" vs "Quantiles of Dist B".
  - Perfect line shown at 45°.
  - Add caption: *Straight line → matching distributions.*

---

## Slide 4 — Theoretical vs Theoretical (Same Distribution)
- Title: *QQ-Plot: Normal vs Normal (Theoretical)*
- Visual instruction:
  - Generate quantiles of Normal(0,1) against quantiles of Normal(0,1).
  - Plot points: they fall exactly on line y=x.
  - Overlay diagonal line for clarity.
  - Caption: *Same distributions → perfect straight line.*

---

## Slide 5 — Empirical vs Empirical (Same Distribution)
- Title: *QQ-Plot: Normal vs Normal (Empirical)*
- Visual instruction:
  - Simulate two independent samples of size n=200 from Normal(0,1).
  - Compute sample quantiles.
  - Plot one sample’s quantiles vs the other’s.
  - Points approximate a straight line, with some scatter.
  - Caption: *Same distribution, sample noise causes scatter.*

---

## Slide 6 — Theoretical vs Theoretical (Different Distributions)
- Title: *QQ-Plot: Normal vs Uniform (Theoretical)*
- Visual instruction:
  - Compute quantiles for Normal(0,1) and Uniform(0,1).
  - Plot Normal quantiles vs Uniform quantiles.
  - Curve deviates from diagonal: S-shape.
  - Caption: *Different distributions → systematic curvature.*

---

## Slide 7 — Empirical vs Empirical (Different Distributions)
- Title: *QQ-Plot: Normal vs Exponential (Empirical)*
- Visual instruction:
  - Simulate n=200 from Normal(0,1) and Exponential(1).
  - Compute sample quantiles.
  - Plot Normal sample quantiles vs Exponential sample quantiles.
  - Points curve strongly (skewed tail visible).
  - Caption: *Shows heavy-tail differences between Normal and Exponential.*

---

## Slide 8 — Takeaways
- Title: *Why QQ-Plots Matter*
- Content (bullets):
  - Tool to check if a dataset matches a theoretical distribution (empirical vs theoretical).
  - Tool to compare two datasets (empirical vs empirical).
  - Straight line → matching distributions.
  - Curvature or deviation → differences in location, spread, or tails.
- Visual instruction:
  - Four small QQ-plot thumbnails (same vs same, empirical vs empirical, different vs different).
  - Caption: *QQ-plots are diagnostics for distributional comparison.*

---

## Notes for the Coding Assistant
- Insert these 8 slides as a consecutive block in the "Descriptive Statistics" section (after histograms/boxplots, before moving to inference concepts).
- Use Beamer `frame` environments consistent with the rest of the file.
- Replace “Visual instruction” placeholders with appropriate `\includegraphics{...}` calls (plots to be generated separately in Python/R/Matlab).
- Equations should be centered, text concise, and bullet-point style consistent with other slides.
