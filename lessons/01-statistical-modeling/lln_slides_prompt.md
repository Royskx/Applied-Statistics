# Agent Prompt: Insert Slides for the Law of Large Numbers (LLN)

You are a LaTeX/Beamer coding assistant. Modify `main.tex` to insert a sequence of slides introducing the Law of Large Numbers. 
These slides should be self-contained, intuitive, and connected to the overall theme of Applied Statistics. 

---

## Slide 1 — Motivation & Intuition
- Title: *Why do averages stabilize?*
- Content:
  - Individual outcomes are random and fluctuate.
  - Averaging across many samples cancels out randomness.
  - The average becomes more predictable as sample size grows.
- Visual placeholder: empirical mean of coin tosses (n=10, 100, 1000) stabilizing around 0.5.

---

## Slide 2 — Visual Demonstration
- Title: *Empirical averages get closer to the true mean*
- Content:
  - Show trajectories of sample averages of Bernoulli(0.5).
  - Early averages fluctuate a lot, later averages stabilize.
  - Phrase: *“The law of large numbers tells us this stabilization is guaranteed.”*

---

## Slide 3 — Informal Statement
- Title: *Law of Large Numbers (Informal)*
- Content:
  - “When we take more and more independent observations, the sample average converges to the true mean.”
  - In words: “Data averages out.”
  - Key insight: this is why we use the sample mean to estimate population parameters.

---

## Slide 4 — Formal Statement (Weak LLN)
- Title: *Law of Large Numbers (Formal)*
- Content:
  - Let X_1, X_2, … be i.i.d. random variables with expectation μ = E[X_1].
  - Define the sample mean: \\( \overline{X}_n = \tfrac{1}{n} \sum_{i=1}^n X_i \\).
  - Then: \\( \overline{X}_n \xrightarrow{\mathbb{P}} \mu \\).
  - Translation: For any ε > 0, P(|\overline{X}_n − μ| > ε) → 0 as n → ∞.

---

## Slide 5 — Takeaways for Statistics
- Title: *Why LLN Matters for Applied Statistics*
- Content:
  - Justifies using sample averages (and estimators) to approximate population parameters.
  - Larger samples ⇒ more stability, less randomness.
  - Basis for estimation theory: more data ⇒ deterministic-looking quantities.
- Visual placeholder: histogram comparison (small sample vs large sample).

---

## Notes for the Coding Assistant
- Insert these 5 slides as a consecutive block in Section 3 (after the slides on independence and before introducing the Central Limit Theorem, if present).
- Wrap each slide with Beamer `frame` environments using consistent style with the rest of the file.
- Use clear bullet points and centered equations.
- Replace all “Visual placeholder” mentions with `\includegraphics{...}` commands pointing to the appropriate image files (to be provided later).
