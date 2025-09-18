# Agent Prompt: Insert Slides for the Central Limit Theorem (CLT)

You are a LaTeX/Beamer coding assistant. Modify `main.tex` to insert a sequence of slides introducing the Central Limit Theorem (CLT). 
These slides should be pedagogical, self-contained, and emphasize both intuition and the formal statement. They should also connect directly to the applied statistics goal of building confidence intervals.

---

## Slide 1 — Motivation & Intuition
- Title: *From stabilization to distributional shape*
- Content:
  - LLN: sample mean \\( \overline{X}_n \\) converges to the true mean \\( \mu \\).
  - CLT: answers *how fast* (scaling by \\( \sqrt{n} \\)) and *with what shape* (Gaussian).
- Visual instruction:
  - Left: plot of a single simulation of coin tosses (0/1 coded, up to n=200). Show \\( \overline{X}_n \\) stabilizing around 0.5.
  - Right: schematic Normal curves centered at \\( \mu \\), variance decreasing as n grows (three curves: wide, medium, narrow).
  - Caption: *LLN = convergence to mean. CLT = Gaussian fluctuations shrinking with 1/√n.*

---

## Slide 2 — Visual Demonstration
- Title: *Coin toss averages become Gaussian*
- Experiment: Toss a fair coin (0/1), repeat many times.
- Visual instruction:
  - Simulate 10,000 repetitions of sample means for n = 5, 20, 50.
  - Compute normalized deviations: \\( Z_n = \sqrt{n}(\overline{X}_n - 0.5) \\).
  - Create histograms for Z_n:
    - Left: n=5 → discrete, skewed shape.
    - Middle: n=20 → smoother, closer to bell curve.
    - Right: n=50 → close to Normal(0,0.25).
  - Overlay each histogram with Normal density curve for comparison.
  - Caption: *As n grows, Z_n looks increasingly Gaussian.*

---

## Slide 3 — Informal Statement
- Title: *Central Limit Theorem (Informal)*
- Content:
  - “LLN says averages stabilize. CLT says that if we zoom in at the scale √n, fluctuations are Gaussian.”
  - Key insight: CLT provides both speed and shape of convergence.
- Visual instruction:
  - Cartoon with three magnifications:
    - Raw averages (points scattered around μ).
    - Zoom scaled by √n → scatter looks bell-shaped.
    - Draw Gaussian bell overlay.
  - Caption: *CLT = shape of convergence.*

---

## Slide 4 — Formal Statement
- Title: *Central Limit Theorem (Formal)*
- Content:
  - Let X_1, X_2, … i.i.d. with mean μ, variance σ² < ∞.
  - Sample mean: \\( \overline{X}_n = \tfrac{1}{n} \sum_{i=1}^n X_i \\).
  - Then: \\( \sqrt{n}\,\tfrac{\overline{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0,1) \\).
  - Interpretation: fluctuations shrink at rate 1/√n and are Gaussian-shaped.
- Visual instruction:
  - Left: distributions of \\( \overline{X}_n \\) for n small/medium/large (curves narrower as n grows).
  - Right: normalized version converging to fixed standard Normal curve.
  - Arrows showing “scaling by √n” leads to stable bell curve.

---

## Slide 5 — Practical Consequences
- Title: *Why the CLT matters for Applied Statistics*
- Content:
  - Approximation: \\( \overline{X}_n \approx \mathcal{N}(\mu, \sigma^2/n) \\).
  - Larger n → smaller variance → narrower confidence intervals.
  - CLT underpins confidence intervals and hypothesis tests.
- Visual instruction:
  - Plot of Normal curves for \\( \overline{X}_n \\) with same mean μ but variances σ²/n shrinking (n=10, 50, 200).
  - Illustration: confidence interval around μ shrinking as n grows (three nested intervals).

---

## Slide 6 — Takeaways
- Title: *Key Insights from the CLT*
- Content (bullets):
  - LLN: averages converge to μ.
  - CLT: fluctuations shrink at rate 1/√n and are Gaussian-shaped.
  - This dual perspective makes inference possible.
- Visual instruction:
  - Split panel:
    - Left: LLN visual (empirical mean trajectory stabilizing).
    - Right: CLT visual (bell curves shrinking, normalized version fixed at N(0,1)).
  - Caption: *LLN = destination. CLT = speed + shape of convergence.*

---

## Notes for the Coding Assistant
- Insert these 6 slides as a consecutive block in Section 3 (after the LLN slides, before introducing confidence intervals).
- Use consistent Beamer `frame` style with the rest of the file.
- Placeholders for visuals should be implemented using `\includegraphics{...}` or TikZ diagrams, depending on style used elsewhere.
- Ensure equations are centered, bullet points concise, and fonts consistent.
