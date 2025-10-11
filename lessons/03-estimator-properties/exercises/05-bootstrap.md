# Exercises: Bootstrap Methods

## Exercise 1: Bootstrap CI for Median of Exponential Data

**Problem:** Use bootstrap to construct a 95% confidence interval for the median of exponential data.

**Solution:**

### Why Bootstrap for the Median?

For the **median**, traditional parametric methods are complex:
- No simple closed-form distribution
- Depends on density at the median
- Asymptotic distribution: $\sqrt{n}(M_n - m) \xrightarrow{d} \mathcal{N}(0, 1/(4f(m)^2))$

Bootstrap provides a simple alternative!

Full bootstrap implementation, diagnostics, and plots are available in the notebook `../notebooks/05-bootstrap.ipynb` under “Exercise 1: Bootstrap CI for the Median (Exponential Data).”

### Key Findings:

1. **Percentile Bootstrap:** Simple and intuitive
2. **Basic Bootstrap:** Better theoretical properties
3. **BCa Bootstrap:** Bias and skewness corrected
4. **All methods:** Provide reasonable CIs without parametric assumptions

### When to Use Bootstrap for Median:

✅ **Advantages:**
- No parametric assumptions
- Works for any distribution
- Simple to implement
- Handles skewness naturally

❌ **Limitations:**
- Requires sufficient sample size (n ≥ 30)
- Computationally intensive
- May be unstable for very small samples

---

## Exercise 2: Bootstrap vs Welch's t-Interval

**Problem:** Compare bootstrap CI for mean difference vs Welch's t-interval using heights data from `shared/data/heights_weights_sample.csv`.

All calculations and visualisations comparing bootstrap and Welch intervals are implemented in `../notebooks/05-bootstrap.ipynb` under “Exercise 2: Bootstrap vs Welch's t-Interval.”

### Comparison Summary:

| Aspect | Welch's t | Bootstrap |
|--------|-----------|-----------|
| **Assumptions** | Normality (robust to violations) | None (non-parametric) |
| **Speed** | Fast (analytical) | Slower (resampling) |
| **Sample size** | Works for small n | Needs n ≥ 20-30 |
| **Heavy tails** | Moderately robust | Very robust |
| **Skewness** | Assumes symmetry | Handles naturally |

**Recommendation:**
- **Normal-ish data:** Welch's t is fine and faster
- **Non-normal data:** Bootstrap is more reliable
- **Small samples:** Welch's t preferred (bootstrap unstable)
- **Large samples:** Both give similar results

---

## Exercise 3: Studentized Bootstrap

**Problem:** Implement studentized bootstrap for the sample mean and compare to percentile bootstrap.

**Solution:**

### Studentized Bootstrap

Instead of bootstrapping the statistic directly, bootstrap the **studentized statistic**:
$$T^* = \frac{\bar{X}^* - \bar{X}}{s^*/\sqrt{n}}$$

This accounts for variability in the standard error.

The full studentized vs percentile bootstrap workflow (including coverage simulation and plots) is provided in `../notebooks/05-bootstrap.ipynb` under “Exercise 3: Studentized Bootstrap for the Mean.”

### Key Findings:

1. **Percentile Bootstrap:** Simple but may have coverage issues for skewed data
2. **Studentized Bootstrap:** Better coverage, especially for skewed distributions
3. **Trade-off:** Studentized is more computationally intensive (nested bootstrap)

**When to use Studentized Bootstrap:**
- Skewed data
- Small to moderate sample sizes
- When coverage accuracy is critical

---

## Exercise 4: Bootstrap Performance with Small Samples

**Problem:** Investigate how bootstrap performance degrades with very small sample sizes ($n = 5, 10$).

**Solution:**

The simulation that benchmarks bootstrap CIs at very small sample sizes (including coverage plots) now runs from the notebook section “Exercise 4: Bootstrap Performance with Small Samples.”

### Key Insights:

1. **n = 5:** Bootstrap is unreliable
   - Coverage can be significantly off
   - High variability in CI width
   - Too few unique resamples

2. **n = 10:** Bootstrap is marginal
   - Coverage slightly below nominal
   - Use with caution
   - Consider parametric alternatives

3. **n ≥ 20:** Bootstrap becomes reliable
   - Coverage approaches nominal level
   - CI widths stabilize

4. **n ≥ 30:** Bootstrap is fully reliable
   - Excellent coverage
   - Minimal bias

**Practical Guidelines:**
- **n < 10:** Avoid bootstrap, use parametric methods
- **10 ≤ n < 20:** Bootstrap with caution, validate assumptions
- **n ≥ 20:** Bootstrap is generally safe
- **n ≥ 30:** Bootstrap preferred for non-normal data

**Why Small Samples Are Problematic:**
- Limited diversity in bootstrap samples
- Discrete empirical distribution
- Finite sample bias
- Cannot approximate tail behavior well
