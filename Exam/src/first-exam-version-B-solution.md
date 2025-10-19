# First Exam — Solutions and Grading Notes (Version B)

Point allocations align with `first-exam-version-B.md`. Equivalent reasoning or algebraically identical answers receive full credit.

---

## Question 1 — Probability Foundations (25 points)

1. **Probability space (6 pts)**
   - Sample space: $\Omega = \{(i,j) : i,j \in \{1,2,3,4,5,6\}\}$, consisting of 36 ordered pairs.
   - Natural $\sigma$-algebra: $\mathcal{F} = 2^{\Omega}$.
   - Probability measure: For the fair die, $P(\omega) = \frac{1}{36}$ for each $\omega \in \Omega$.
   Partial credit: correct $\Omega$ (2 pts), acceptable $\mathcal{F}$ (2 pts), correct probabilities summing to 1 (2 pts).

2. **Event probabilities (6 pts)**
   - $A$ = {sum $\ge$ 10} = $\{(4,6), (5,5), (5,6), (6,4), (6,5), (6,6)\}$, so $P(A) = \frac{6}{36} = \frac{1}{6}$.
   - $B$ = {first roll is 5 or 6} = $\{(5,j), (6,j) : j \in \{1,2,3,4,5,6\}\}$, so $P(B) = \frac{12}{36} = \frac{1}{3}$.
   - $A \cap B = \{(5,5), (5,6), (6,4), (6,5), (6,6)\}$, so $P(A \cap B) = \frac{5}{36}$.
   Award 3 pts for correct event descriptions, 3 pts for the resulting probabilities.

3. **Independence check (5 pts)**
   - $P(A)P(B) = \frac{1}{6} \cdot \frac{1}{3} = \frac{1}{18} = \frac{2}{36} \neq \frac{5}{36} = P(A \cap B)$.
   - Therefore $A$ and $B$ are **not independent**.
   Credit: 3 pts for the product comparison, 2 pts for the explicit conclusion.

4. **Biased die analysis (8 pts)**
   - With outcomes 1-5 equally likely with total probability $1-q$, each has probability $\frac{1-q}{5}$.
   - Outcome 6 has probability $q$.
   - Event $B$ (first roll is 5 or 6) has probability:
     $$P_q(B) = \frac{1-q}{5} + q = \frac{1-q + 5q}{5} = \frac{1 + 4q}{5}.$$
   - Setting $P_q(B) = \frac{1}{3}$:
     $$\frac{1 + 4q}{5} = \frac{1}{3} \implies 3(1 + 4q) = 5 \implies 3 + 12q = 5 \implies 12q = 2 \implies q = \frac{1}{6}.$$
   - However, we are told $0 < q < \frac{1}{6}$, so there is **no valid solution** in the given range. (Accept $q = \frac{1}{6}$ as the boundary case if students note the constraint.)
   Award 5 pts for correct expression for $P_q(B)$, 3 pts for solving the equation and noting the constraint issue.

---

## Question 2 — Bayes' Rule in Fraud Detection (25 points)

Let $F$ denote "fraudulent" and $+$ denote "flagged." Given $P(F)=\tfrac{1}{50}$, $P(+\mid F)=\tfrac{4}{5}$, and $P(-\mid F^c)=\tfrac{39}{40}$, so $P(+\mid F^c)=1-\tfrac{39}{40}=\tfrac{1}{40}$.

1. **Probability of a flag (7 pts)**
   $$P(+) = P(+\mid F)P(F) + P(+\mid F^c)P(F^c) = \frac{4}{5}\cdot\frac{1}{50} + \frac{1}{40}\cdot\frac{49}{50} = \frac{16}{1000} + \frac{49}{2000} = \frac{81}{2000}.$$
   Full credit for exact fraction $\tfrac{81}{2000}$.

2. **Positive predictive value (8 pts)**
   $$P(F\mid +) = \frac{P(+\mid F)P(F)}{P(+)} = \frac{\frac{4}{5}\cdot\frac{1}{50}}{\frac{81}{2000}} = \frac{\frac{16}{1000}}{\frac{81}{2000}} = \frac{32}{81}.$$

3. **Posterior after a negative (6 pts)**
   - $P(-) = P(-\mid F^c)P(F^c) + P(-\mid F)P(F) = \frac{39}{40}\cdot\frac{49}{50} + \frac{1}{5}\cdot\frac{1}{50} = \frac{1919}{2000}$.
   - Hence
     $$P(F\mid -) = \frac{P(-\mid F)P(F)}{P(-)} = \frac{\frac{1}{5}\cdot\frac{1}{50}}{\frac{1919}{2000}} = \frac{1/250}{1919/2000} = \frac{8}{1919}.$$

4. **General formula and interpretation (4 pts)**
   - With $\pi=P(F)$, $s=P(+\mid F)$ and $c=P(-\mid F^c)$:
     $$P(F\mid +) = \frac{s\pi}{s\pi + (1-c)(1-\pi)}.$$
   - Increasing $c$ decreases $(1-c)$, reducing the denominator and thereby increasing $P(F\mid +)$.
   Credit: 3 pts for correct formula, 1 pt for the qualitative explanation.

---

## Question 3 — Uniform Waiting-Time Model (25 points)

1. **CDF (5 pts)**
   For $T \sim \mathrm{Uniform}(0,8)$,
   $$F_T(t) =
   \begin{cases}
   0, & t < 0,\\
   \frac{t}{8}, & 0 \le t \le 8,\\
   1, & t > 8.
   \end{cases}$$

2. **Probability of an interval (5 pts)**
   $$P(2 \le T \le 6) = F_T(6) - F_T(2) = \frac{6}{8} - \frac{2}{8} = \frac{4}{8} = \frac{1}{2}.$$

3. **Expectation and variance via integration (9 pts)**
   - Density: $f_T(t) = \frac{1}{8}$ for $t\in[0,8]$.
   - Expectation:
     $$E[T] = \int_0^8 t \cdot \frac{1}{8}\, dt = \frac{1}{8} \cdot \frac{t^2}{2}\Big|_0^8 = \frac{1}{8} \cdot 32 = 4.$$
   - Second moment:
     $$E[T^2] = \int_0^8 t^2 \cdot \frac{1}{8}\, dt = \frac{1}{8} \cdot \frac{t^3}{3}\Big|_0^8 = \frac{1}{8} \cdot \frac{512}{3} = \frac{64}{3}.$$
   - Variance: $\mathrm{Var}(T) = E[T^2] - (E[T])^2 = \frac{64}{3} - 16 = \frac{16}{3}$.

4. **Affine transformation (6 pts)**
   - $E[Y] = E[2T + 5] = 2E[T] + 5 = 2 \cdot 4 + 5 = 13$.
   - $\mathrm{Var}(Y) = 2^2 \mathrm{Var}(T) = 4 \cdot \frac{16}{3} = \frac{64}{3}$.
   Full credit for applying linearity rules correctly.

---

## Question 4 — Interpreting Summaries and Convergence (25 points)

1. **Law of Large Numbers interpretation (9 pts)**
   - The average of the sample means in Table 1 stays close to the true mean 3: $3.06$ for $n=10$, $3.01$ for $n=40$, and $3.00$ for $n=160$.
   - As $n$ increases, the averages move closer to 3 and fluctuate less, illustrating that sample means converge toward the population mean, as predicted by the Law of Large Numbers.
   - Key elements for full credit: reference to the table values, explicit connection to convergence of $\bar{X}_n$ toward $\mu$.

2. **Central Limit Theorem interpretation (8 pts)**
   - The standard deviation of the 800 sample means decreases roughly by a factor close to $1/\sqrt{n}$: $0.94$ for $n=10$, $0.47$ for $n=40$, and $0.24$ for $n=160$.
   - Note that $0.94/\sqrt{10} \approx 0.30$, $0.94/\sqrt{40} \approx 0.15$, and $0.94/\sqrt{160} \approx 0.074$, showing the approximate $1/\sqrt{n}$ scaling (allowing for simulation variability).
   - This shrinking variability reflects the CLT prediction that $\bar{X}_n$ becomes more concentrated and approximately normal with variance $\sigma^2 / n$ as $n$ grows.
   - Full credit requires explicitly linking the observed reductions to the CLT scaling.

3. **QQ-plot assessment and recommendation (8 pts)**
   - Sample quantiles are less extreme in both tails than the normal quantiles (e.g., $-1.4 > -2$ and $1.6 < 2$), indicating a QQ-plot that curves toward the center with lighter tails than normal.
   - Therefore the dataset appears to have **lighter tails** than a standard normal distribution.
   - Suggested actions: the data may be closer to uniform or bounded; consider checking for truncation, using a bounded distribution model, or investigating if normal-based inference is still appropriate given the lighter tails.
   - Full credit needs description of the pattern, a clear normality conclusion, and an appropriate recommendation.

---

**Grading tip:** Emphasise exact algebra, clear reasoning, and structured explanations when awarding partial credit.
