# First Exam — Lessons 0 & 1 (Version A)

**Time allowed:** 60 minutes
**Resources:** Printed course notes and personal handwritten notes only. Electronic devices, calculators, and statistical tables are not permitted.
**Instructions:** Answer all four questions. Show algebraic steps and clearly label each part. Unless stated otherwise, provide exact values (fractions, radicals, exponentials). Partial credit is available when reasoning is clearly explained.

| Question | Topic | Points |
|----------|-------|--------|
| 1 | Probability structures and independence | 25 |
| 2 | Bayes' rule and conditional reasoning | 25 |
| 3 | Exponential model derivations | 25 |
| 4 | Interpreting summaries and convergence | 25 |
| **Total** |  | **100** |

---

## Question 1 — Probability Foundations (25 points)

A coin is tossed twice. Throughout parts (a)–(c), assume the coin is fair.

1. (**6 pts**) Specify a probability space $(\Omega, \mathcal{F}, P)$ for this experiment. Clearly list the sample space, a natural $\sigma$-algebra, and the probability measure.
2. (**6 pts**) Let $A$ be the event "at least one head appears" and $B$ the event "exactly one head appears." Compute $P(A)$, $P(B)$, and $P(A \cap B)$.
3. (**5 pts**) Are $A$ and $B$ independent? Justify your answer using the formal definition.
4. (**8 pts**) Now suppose the coin has probability $p$ of landing heads, where $0 < p < 1$. With the same definitions of $A$ and $B$, derive expressions for $P_p(A)$, $P_p(B)$, and $P_p(A \cap B)$. For which value(s) of $p$ are $A$ and $B$ independent? Provide algebraic reasoning.

---

## Question 2 — Bayes' Rule in Component Testing (25 points)

A factory purchases circuit boards from two suppliers. Boards from $S_A$ account for $\frac{3}{5}$ of the total, and boards from $S_B$ account for the remaining $\frac{2}{5}$. Historical quality data show that $\frac{1}{10}$ of boards from $S_A$ are defective, while $\frac{1}{20}$ of boards from $S_B$ are defective. Each board is screened by a test that returns "positive" for a defective board with probability $\frac{9}{10}$, and returns "positive" for a non-defective board with probability $\frac{1}{20}$.

Recall: sensitivity $= P(\text{Positive} \mid \text{Defective})$ and specificity $= P(\text{Negative} \mid \text{Non\text{-}defective})$. The false positive rate is $1-\text{specificity} = P(\text{Positive} \mid \text{Non\text{-}defective})$.

1. (**7 pts**) Compute the probability that a randomly selected board produces a positive test result.
2. (**8 pts**) Given a positive test result, determine the probability that the board is defective. Express your answer as a reduced fraction.
3. (**6 pts**) Given a negative test result, determine the probability that the board is defective. Express your answer as a reduced fraction.
4. (**4 pts**) Let $\pi = P(\text{Defective})$, $s = P(\text{Positive} \mid \text{Defective})$, and $f = P(\text{Positive} \mid \text{Non-defective})$. Derive a formula for $P(\text{Defective} \mid \text{Positive})$ in terms of $\pi$, $s$, and $f$, and briefly state how decreasing $f$ affects this probability.

---

## Question 3 — Exponential Waiting-Time Model (25 points)

Let $T$ be the waiting time (in hours) until the next service request arrives, and suppose $T$ follows an exponential distribution with rate $\lambda = \frac{1}{4}$.

**Recall:** The probability density function of an exponential random variable with rate $\lambda$ is given by
$$f_T(t) = \begin{cases}
\lambda e^{-\lambda t}, & t \ge 0,\\
0, & t < 0.
\end{cases}$$

1. (**5 pts**) Derive the cumulative distribution function $F_T(t)$.
2. (**5 pts**) Using your result from part (1), compute $P(1 \le T \le 4)$ and leave the answer in exponential form.
3. (**9 pts**) Compute $E[T]$ and $\mathrm{Var}(T)$ directly from integrals. Show the essential integration steps leading to exact values.
4. (**6 pts**) Define $Y = 3T + 2$. Compute $E[Y]$ and $\mathrm{Var}(Y)$ using properties of expectation and variance. Provide exact values.

---

## Question 4 — Interpreting Summaries and Convergence (25 points)

Table 1 summarises the results of a simulation: for each sample size $n$, 1,000 independent samples of size $n$ were drawn from an exponential distribution with mean $4$, and the sample mean was recorded for each run.

**Table 1 — Behaviour of Sample Means**

| Sample size $n$ | Average of the 1,000 sample means | Standard deviation of the 1,000 sample means |
|-----------------|-----------------------------------|----------------------------------------------|
| 5               | 4.08                              | 1.80                                         |
| 20              | 4.02                              | 0.90                                         |
| 80              | 4.01                              | 0.44                                         |

Another analysis compared a standardised dataset (mean 0, variance 1) to the standard normal distribution. Selected points from the QQ-plot are listed below.

**Table 2 — Selected QQ-Plot Quantiles**

| Normal quantile | Sample quantile |
|-----------------|-----------------|
| -2.0            | -3.1            |
| -1.0            | -1.6            |
| 0.0             | 0.2             |
| 1.0             | 1.9             |
| 2.0             | 3.4             |

Answer the following:

1. (**9 pts**) Use Table 1 to explain how the Law of Large Numbers manifests in this simulation. Refer explicitly to the numerical values.
2. (**8 pts**) Still relying on Table 1, discuss how the variability of the sample mean changes with $n$ and connect your explanation to the Central Limit Theorem.
3. (**8 pts**) Interpret Table 2. Describe the shape of the corresponding QQ-plot, assess the plausibility of a normal model, and recommend a modelling or diagnostic action before relying on normal-based methods.

---

**End of exam.** Ensure answers are clearly labelled and presented legibly.
