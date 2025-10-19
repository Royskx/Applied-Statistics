# First Exam — Solutions and Grading Notes (Version A)

Point allocations align with `first-exam-version-A.md`. Equivalent reasoning or algebraically identical answers receive full credit.

---

## Question 1 — Probability Foundations (25 points)

1. **Probability space (6 pts)**
   - Sample space: $\Omega = \{\text{HH}, \text{HT}, \text{TH}, \text{TT}\}$.
   - Natural $\sigma$-algebra: $\mathcal{F} = 2^{\Omega}$.
   - Probability measure: For the fair coin, $P(\omega) = \frac{1}{4}$ for each $\omega \in \Omega$.
   Partial credit: correct $\Omega$ (2 pts), acceptable $\mathcal{F}$ (2 pts), correct probabilities summing to 1 (2 pts).

2. **Event probabilities (6 pts)**
   - $A = \{\text{HH}, \text{HT}, \text{TH}\}$ so $P(A) = \frac{3}{4}$.
   - $B = \{\text{HT}, \text{TH}\}$ so $P(B) = \frac{1}{2}$.
   - $A \cap B = \{\text{HT}, \text{TH}\}$ so $P(A \cap B) = \frac{1}{2}$.
   Award 3 pts for correct event descriptions, 3 pts for the resulting probabilities.

3. **Independence check (5 pts)**
   - $P(A)P(B) = \frac{3}{4} \cdot \frac{1}{2} = \frac{3}{8} \neq \frac{1}{2} = P(A \cap B)$.
   - Therefore $A$ and $B$ are **not independent**.
   Credit: 3 pts for the product comparison, 2 pts for the explicit conclusion.

4. **Biased coin analysis (8 pts)**
   - With head probability $p$, $P_p(\text{HH}) = p^2$, $P_p(\text{HT}) = p(1-p)$, $P_p(\text{TH}) = (1-p)p$, $P_p(\text{TT}) = (1-p)^2$.
   - $P_p(A) = 1 - (1-p)^2 = 2p - p^2$.
   - $P_p(B) = 2p(1-p)$.
   - $P_p(A \cap B) = P_p(B) = 2p(1-p)$.
   - Independence would require $P_p(A \cap B) = P_p(A)P_p(B)$, so
     $$2p(1-p) = (2p - p^2)(2p(1-p)).$$
     If $0 < p < 1$, then $P_p(B) > 0$ and division by $2p(1-p)$ yields $1 = 2p - p^2$, i.e. $(p-1)^2 = 0$. The only solution is $p = 1$, but that value lies outside $(0,1)$ and renders $B$ impossible. Hence for $0 < p < 1$, $A$ and $B$ are not independent.
   Award 5 pts for correct expressions, 3 pts for the independence argument.

---

## Question 2 — Bayes' Rule in Component Testing (25 points)

Let $D$ denote "defective" and $+$ denote "positive test."

1. **Overall positive probability (7 pts)**
   - Prevalence:
     $$P(D) = \frac{3}{5} \cdot \frac{1}{10} + \frac{2}{5} \cdot \frac{1}{20} = \frac{3}{50} + \frac{1}{50} = \frac{2}{25}.$$
   - Therefore
     $$P(+) = P(+ \mid D)P(D) + P(+ \mid D^c)P(D^c) = \frac{9}{10} \cdot \frac{2}{25} + \frac{1}{20} \cdot \frac{23}{25} = \frac{36}{500} + \frac{23}{500} = \frac{59}{500}.$$
   Full credit for exact fraction $\frac{59}{500}$.

2. **Positive predictive value (8 pts)**
   $$P(D \mid +) = \frac{P(+ \mid D)P(D)}{P(+)} = \frac{\frac{9}{10} \cdot \frac{2}{25}}{\frac{59}{500}} = \frac{\frac{9}{125}}{\frac{59}{500}} = \frac{36}{59}.$$
   Reduced fraction $\frac{36}{59}$ earns full credit.

3. **Posterior after a negative (6 pts)**
   - $P(- \mid D) = 1 - \frac{9}{10} = \frac{1}{10}$, $P(- \mid D^c) = 1 - \frac{1}{20} = \frac{19}{20}$.
   - $P(-) = \frac{1}{10} \cdot \frac{2}{25} + \frac{19}{20} \cdot \frac{23}{25} = \frac{1}{125} + \frac{437}{500} = \frac{441}{500}$.
   - Hence
     $$P(D \mid -) = \frac{\frac{1}{10} \cdot \frac{2}{25}}{\frac{441}{500}} = \frac{\frac{1}{125}}{\frac{441}{500}} = \frac{4}{441}.$$

4. **General formula and interpretation (4 pts)**
   - Bayes' rule gives
     $$P(D \mid +) = \frac{s\pi}{s\pi + f(1-\pi)}.$$
   - Decreasing $f$ (the false-positive rate) reduces the denominator while leaving the numerator unchanged, thereby increasing $P(D \mid +)$.
   Credit: 3 pts for correct formula, 1 pt for the qualitative explanation.

---

## Question 3 — Exponential Waiting-Time Model (25 points)

1. **CDF (5 pts)**
   For $\lambda = \frac{1}{4}$,
   $$F_T(t) =
   \begin{cases}
   0, & t < 0,\\
   1 - e^{-t/4}, & t \ge 0.
   \end{cases}$$

2. **Probability of an interval (5 pts)**
   $$P(1 \le T \le 4) = F_T(4) - F_T(1) = (1 - e^{-1}) - (1 - e^{-1/4}) = e^{-1/4} - e^{-1}.$$

3. **Expectation and variance via integration (9 pts)**
   - Expectation:
     $$E[T] = \int_0^\infty t \cdot \frac{1}{4} e^{-t/4}\, dt.$$
     Using integration by parts with $u = t$ and $dv = \frac{1}{4} e^{-t/4} dt$ (or substituting $x = t/4$) yields $E[T] = 4$.
   - Second moment:
     $$E[T^2] = \int_0^\infty t^2 \cdot \frac{1}{4} e^{-t/4} \, dt = 16 \int_0^\infty x^2 e^{-x} dx = 16 \cdot 2! = 32,$$
     where $x = t/4$ and the gamma integral $\int_0^\infty x^2 e^{-x} dx = 2!$ is used (students may perform repeated integration by parts).
   - Variance: $\mathrm{Var}(T) = E[T^2] - (E[T])^2 = 32 - 16 = 16$.

4. **Affine transformation (6 pts)**
   - $E[Y] = E[3T + 2] = 3E[T] + 2 = 3 \cdot 4 + 2 = 14$.
   - $\mathrm{Var}(Y) = 3^2 \mathrm{Var}(T) = 9 \cdot 16 = 144$.
   Full credit for applying linearity rules correctly.

---

## Question 4 — Interpreting Summaries and Convergence (25 points)

1. **Law of Large Numbers interpretation (9 pts)**
   - The average of the sample means in Table 1 stays close to the true mean 4: $4.08$ for $n=5$, $4.02$ for $n=20$, and $4.01$ for $n=80$.
   - As $n$ increases, the averages move closer to 4 and fluctuate less, illustrating that sample means converge toward the population mean, as predicted by the Law of Large Numbers.
   - Key elements for full credit: reference to the table values, explicit connection to convergence of $\bar{X}_n$ toward $\mu$.

2. **Central Limit Theorem interpretation (8 pts)**
   - The standard deviation of the 1,000 sample means decreases roughly by a factor close to $1/\sqrt{n}$: $1.80$ for $n=5$, $0.90$ for $n=20$, and $0.44$ for $n=80$.
   - This shrinking variability reflects the CLT prediction that $\bar{X}_n$ becomes more concentrated and approximately normal with variance $\sigma^2 / n$ as $n$ grows.
   - Full credit requires explicitly linking the observed reductions to the CLT scaling.

3. **QQ-plot assessment and recommendation (8 pts)**
   - Sample quantiles are more extreme in both tails than the normal quantiles (e.g., $-3.1 < -2$ and $3.4 > 2$), indicating a QQ-plot that bends away from the 45° line with heavier tails.
   - Therefore the dataset is unlikely to follow a standard normal distribution closely.
   - Suggested actions: adopt a heavier-tailed model (e.g., Student-$t$), employ robust procedures, or transform the data before applying normal-based methods.
   - Full credit needs description of the pattern, a clear normality conclusion, and an appropriate recommendation.

---

**Grading tip:** Emphasise exact algebra, clear reasoning, and structured explanations when awarding partial credit.
