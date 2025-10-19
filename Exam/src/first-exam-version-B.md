# First Exam — Lessons 0 & 1 (Version B)

**Time allowed:** 60 minutes
**Resources:** Printed course notes and personal handwritten notes only. Electronic devices, calculators, and statistical tables are not permitted.
**Instructions:** Answer all four questions. Show algebraic steps and clearly label each part. Unless stated otherwise, provide exact values (fractions, radicals, exponentials). Partial credit is available when reasoning is clearly explained.

| Question | Topic | Points |
|----------|-------|--------|
| 1 | Probability structures and independence | 25 |
| 2 | Bayes' rule and conditional reasoning | 25 |
| 3 | Uniform model derivations | 25 |
| 4 | Interpreting summaries and convergence | 25 |
| **Total** |  | **100** |

---

## Question 1 — Probability Foundations (25 points)

A die is rolled twice. Throughout parts (a)–(c), assume the die is fair.

1. (**6 pts**) Specify a probability space $(\Omega, \mathcal{F}, P)$ for this experiment. Clearly describe the sample space, a natural $\sigma$-algebra, and the probability measure.
2. (**6 pts**) Let $A$ be the event "the sum of the two rolls is at least 10" and $B$ the event "the first roll shows a 5 or 6." Compute $P(A)$, $P(B)$, and $P(A \cap B)$.
3. (**5 pts**) Are $A$ and $B$ independent? Justify your answer using the formal definition.
4. (**8 pts**) Now suppose the die is biased such that the probability of rolling a 6 is $q$ (where $0 < q < \frac{1}{6}$), while outcomes 1 through 5 are equally likely with total probability $1-q$. With the same definition of event $B$, derive an expression for $P_q(B)$ in terms of $q$. For which value of $q$ does $P_q(B) = \frac{1}{3}$? Provide algebraic reasoning.

---

## Question 2 — Bayes' Rule in Fraud Detection (25 points)

A bank uses an automated system to flag potentially fraudulent credit card transactions. Historical data indicate that $\tfrac{1}{50}$ of transactions are actually fraudulent. The system has sensitivity (true positive rate) $\tfrac{4}{5}$ and specificity (true negative rate) $\tfrac{39}{40}$.

Recall: sensitivity $= P(\text{Flag} \mid \text{Fraud})$ and specificity $= P(\text{No Flag} \mid \text{No Fraud})$. The false positive rate is $1-\text{specificity} = P(\text{Flag} \mid \text{No Fraud})$.

1. (**7 pts**) Compute the probability that a randomly selected transaction is flagged as suspicious.
2. (**8 pts**) Given that a transaction is flagged, determine the probability that it is actually fraudulent. Express your answer as a reduced fraction.
3. (**6 pts**) Given that a transaction is not flagged, determine the probability that it is actually fraudulent. Express your answer as a reduced fraction.
4. (**4 pts**) Let $\pi = P(\text{Fraud})$, $s = P(\text{Flag} \mid \text{Fraud})$ (sensitivity), and $c = P(\text{No Flag} \mid \text{No Fraud})$ (specificity). Derive a formula for $P(\text{Fraud} \mid \text{Flag})$ in terms of $\pi$, $s$, and $c$, and briefly state how increasing $c$ affects this probability.

---

## Question 3 — Uniform Waiting-Time Model (25 points)

Let $T$ be the time (in minutes) until the next customer arrives at a service counter, and suppose $T$ follows a uniform distribution on $[0,8]$.

**Recall:** The probability density function of a $\mathrm{Uniform}(a,b)$ random variable is
$$f_T(t) = \begin{cases}
\dfrac{1}{b-a}, & a \le t \le b,\\
0, & \text{otherwise},
\end{cases}$$
and the cumulative distribution function is $F_T(t) = 0$ for $t<a$, $F_T(t) = \dfrac{t-a}{b-a}$ for $a \le t \le b$, and $F_T(t) = 1$ for $t>b$.

1. (**5 pts**) Specialise the CDF $F_T(t)$ for $T \sim \mathrm{Uniform}(0,8)$.
2. (**5 pts**) Using your result from part (1), compute $P(2 \le T \le 6)$.
3. (**9 pts**) Compute $E[T]$ and $\mathrm{Var}(T)$ directly from integrals. Show the essential integration steps leading to exact values.
4. (**6 pts**) Define $Y = 2T + 5$. Compute $E[Y]$ and $\mathrm{Var}(Y)$ using properties of expectation and variance. Provide exact values.

---

## Question 4 — Interpreting Summaries and Convergence (25 points)

Table 1 summarises the results of a simulation: for each sample size $n$, 800 independent samples of size $n$ were drawn from an exponential distribution with mean $3$, and the sample mean was recorded for each run.

**Table 1 — Behaviour of Sample Means**

| Sample size $n$ | Average of the 800 sample means | Standard deviation of the 800 sample means |
|-----------------|----------------------------------|---------------------------------------------|
| 10              | 3.06                             | 0.94                                        |
| 40              | 3.01                             | 0.47                                        |
| 160             | 3.00                             | 0.24                                        |

Another analysis compared a standardised dataset (mean 0, variance 1) to the standard normal distribution. Selected points from the QQ-plot are listed below.

**Table 2 — Selected QQ-Plot Quantiles**

| Normal quantile | Sample quantile |
|-----------------|-----------------|
| -2.0            | -1.4            |
| -1.0            | -0.8            |
| 0.0             | 0.1             |
| 1.0             | 0.9             |
| 2.0             | 1.6             |

Answer the following:

1. (**9 pts**) Use Table 1 to explain how the Law of Large Numbers manifests in this simulation. Refer explicitly to the numerical values.
2. (**8 pts**) Still relying on Table 1, discuss how the variability of the sample mean changes with $n$ and connect your explanation to the Central Limit Theorem.
3. (**8 pts**) Interpret Table 2. Describe the shape of the corresponding QQ-plot, assess the plausibility of a normal model, and recommend a modelling or diagnostic action before relying on normal-based methods.

---

**End of exam.** Ensure answers are clearly labelled and presented legibly.
