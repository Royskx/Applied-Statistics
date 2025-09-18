# Agent Prompt: Insert a “Descriptive vs Inferential Statistics” slide at the start of Section 3 (“Moments and Convergence”) in `main.tex`

You are a LaTeX/Beamer coding assistant. Modify the file `main.tex` **in place** to insert a single introductory slide on **descriptive statistics** (with a contrast to inferential statistics) **at the very beginning of Section 3 titled `Moments and Convergence`**.

> ✅ Do **not** include any animations or external images.  
> ✅ Keep the current Beamer theme and preamble as-is (no new packages unless strictly necessary).  
> ✅ Make the edit **idempotent**: if the slide was already inserted (markers present), replace it in place rather than duplicating.

---

## 1) Make a safety backup
- Create a copy of `main.tex` named `main.backup.before-descriptive-slide.tex` in the same directory.

---

## 2) Locate Section 3 anchor
Find the section command that starts Section 3, matching any of the following forms (whitespace-insensitive):
- `\section{Moments and Convergence}`
- `\section*{Moments and Convergence}`
- `\section[Moments and Convergence]{Moments and Convergence}`

> If multiple matches exist, use the first occurrence.  
> Insert **immediately after** the section command line (before any other frames of that section).

---

## 3) Idempotent insertion markers
Use these exact comment markers to bound the auto-generated slide:
```tex
% BEGIN AUTO-GENERATED: DescriptiveStatisticsIntro (do not edit)
... slide code ...
% END AUTO-GENERATED: DescriptiveStatisticsIntro
```
- If these markers already exist anywhere **within Section 3**, replace the entire block between them with the new code below (do not insert a second copy).

---

## 4) Insert the slide code
Insert the following Beamer frame **right after** the Section 3 command (or replace the existing marked block if present):

```tex
% BEGIN AUTO-GENERATED: DescriptiveStatisticsIntro (do not edit)
\begin{frame}{Descriptive vs Inferential Statistics}
  \begin{columns}[T,onlytextwidth]
    \begin{column}{0.52\textwidth}
      \textbf{Descriptive statistics}
      \begin{itemize}
        \item \textit{Purpose:} summarize and present the features of the observed dataset; no claims beyond the data.
        \item \textit{Mathematical view:} summary functionals \(S(\mathbf{x})\) of a sample \(\mathbf{x} = (x_1,\dots,x_n)\) capturing location, spread, shape, and dependence.
      \end{itemize}
      \vspace{0.3em}
      \textbf{Examples}
      \begin{itemize}
        \item Mean \(\displaystyle \bar{x}=\frac{1}{n}\sum_{i=1}^n x_i\), median, mode.
        \item Variance \(\displaystyle s^2=\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})^2\), standard deviation, range, IQR.
        \item Counts/proportions (categorical), histograms, boxplots, scatter plots.
      \end{itemize}
    \end{column}
    \begin{column}{0.46\textwidth}
      \textbf{Inferential statistics}
      \begin{itemize}
        \item \textit{Purpose:} generalize from the sample to the population using probability models; quantify uncertainty.
        \item \textit{Outputs:} estimates with standard errors, confidence intervals, hypothesis tests, predictive statements.
      \end{itemize}
      \vspace{0.3em}
      \textbf{Examples}
      \begin{itemize}
        \item Confidence intervals; hypothesis tests (e.g., \(t\)-test); \(p\)-values.
        \item Parametric/GLM/Regression inference; model comparison/selection.
        \item Bootstrap-based uncertainty quantification.
      \end{itemize}
    \end{column}
  \end{columns}
  \medskip
  \textbf{Key distinction:} \emph{descriptive} = “what this dataset looks like”, \emph{inferential} = “what we can say about the population (with uncertainty)”.
\end{frame}
% END AUTO-GENERATED: DescriptiveStatisticsIntro
```

Notes:
- Keep math environments inline and minimal; rely on existing packages from the preamble. If `amsmath` is not loaded and compilation fails, add `\usepackage{amsmath}` in the preamble \emph{only if necessary}.
- Do not modify other frames, sections, or theme settings.

---

## 5) Verify placement and compile
1. Ensure the inserted frame is the **first frame** under Section 3 (“Moments and Convergence”).  
2. Compile `main.tex`.  
3. Confirm the slide appears with the correct title and content, and that navigation (if enabled) reflects the change.

---

## 6) Idempotency check
- Re-running these instructions should not create duplicate slides. The block delimited by the markers must be replaced, not duplicated.

---

## 7) Report back
- If the section anchor was not found, report:  
  `Section 'Moments and Convergence' not found; no changes made.`  
- Otherwise, report success with:  
  `Inserted/Updated 'Descriptive vs Inferential Statistics' frame at the start of Section 3.`
