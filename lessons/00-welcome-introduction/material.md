# Lesson 0: Welcome to Applied Statistics — Course Material

Author: Stéphane Rivaud, INRIA Saclay
Prerequisites: Basic mathematical maturity
Estimated time: 2-3 hours self-study

## Learning Objectives
- Understand the motivation and scope of applied statistics
- Distinguish between probability and statistics as dual frameworks
- Master fundamental probability concepts: events, probability spaces, conditional probability
- Apply Bayes' theorem to update beliefs with new evidence
- Understand independence and its implications
- Connect theoretical concepts to real-world applications

## 1. What are Statistics?

### The Coin Toss Example
Consider a coin toss with unknown probability of heads θ:
- **Probability perspective**: Given θ, what is P(Heads)?
- **Statistics perspective**: Given observed tosses, what is our belief about θ?

This illustrates the **duality** between probability and statistics:
- **Probability**: Forward process (parameters → data)
- **Statistics**: Inverse process (data → parameters)

### Cox-Jaynes Theorem: Quantifying Plausibility

The theorem establishes that any coherent method for quantifying plausibility must be isomorphic to probability theory.

**Requirements for coherence**:
1. **Coherence**: If a result can be derived in multiple ways, all should yield the same answer
2. **Continuity**: Small parameter changes should not cause discontinuous jumps in the method
3. **Universality**: The method should be general, not tied to specific cases

**Requirements for the practitioner**:
1. **Unambiguous specifications**: Propositions must have unique interpretations
2. **No hidden information**: All relevant information must be provided

**Conclusion**: Any coherent reasoning under uncertainty must follow probability theory.

## 2. Key Questions in Applied Statistics

### What is Applied Statistics?
Applied statistics focuses on extracting meaningful insights from data to inform decision-making in real-world contexts.

### Core Questions We Can Answer
- **What is the probability of an event?** (Prediction)
- **What parameter values are most consistent with observed data?** (Estimation)
- **Are two processes different?** (Hypothesis testing)
- **How confident are we in our conclusions?** (Uncertainty quantification)

### The Role of Data vs Theory
- **Data-driven approach**: Let the data speak, but guided by statistical principles
- **Theory-guided approach**: Use domain knowledge to inform models
- **Balance**: Good applied statistics combines both

### Computer Science in Statistics
Modern statistics is computationally intensive because:
- Large datasets require efficient algorithms
- Complex models need numerical optimization
- Simulation-based methods (bootstrap, MCMC) are computationally demanding
- High-dimensional data requires sophisticated techniques

### Statistics vs Machine Learning
- **Statistics**: Focus on inference, uncertainty quantification, causal relationships
- **Machine Learning**: Focus on prediction accuracy, often with less emphasis on interpretability
- **Overlap**: Both use data to learn patterns, but with different emphases

## 3. Real-World Applications

### Gaming and Strategy
- **Casino games**: Understanding house edges and optimal strategies
- **Chess/Go**: Modeling opponent behavior and game theory
- **Network analysis**: Understanding complex system interactions

### Medical Diagnosis
- **Bayesian networks**: Modeling relationships between symptoms and diseases
- **Differential diagnosis**: Systematically ruling out conditions
- **Treatment effectiveness**: Clinical trial design and analysis

### Physics and Scientific Discovery
- **Astronomy**: Signal detection in noisy data
- **Particle physics**: Higgs boson discovery through statistical analysis
- **Experimental design**: Optimizing data collection for maximum information

### Natural Language Processing
- **Machine translation**: Statistical models for language understanding
- **Spam detection**: Classification algorithms with uncertainty quantification
- **Sentiment analysis**: Extracting meaning from text data

## 4. Probability on Sets

### Events and Realizations
- **Sample space Ω**: Set of all possible outcomes
- **Event A**: Subset of Ω (A ⊆ Ω)
- **Realization**: Specific outcome x ∈ Ω that either belongs to A or not

**Examples**:
- Dice rolls: Ω = {1,2,3,4,5,6}
- Chess moves: Ω = set of all legal moves
- Game outcomes: Ω = {win, loss, draw}

### Probability Space
A probability space is a triple (Ω, ℱ, P) where:
- **Ω**: Sample space
- **ℱ**: σ-algebra of events (subsets of Ω that are measurable)
- **P**: Probability measure satisfying:
  1. 0 ≤ P(A) ≤ 1 for any event A
  2. P(Ω) = 1
  3. For disjoint events A₁, A₂, ...: P(∪Aᵢ) = ΣP(Aᵢ)

### Basic Probability Rules
- **Complement**: P(Aᶜ) = 1 - P(A)
- **Union**: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- **Monotonicity**: A ⊆ B implies P(A) ≤ P(B)
- **Bounds**: 0 ≤ P(A) ≤ 1, P(∅) = 0, P(Ω) = 1

## 5. Conditional Probability

### Definition
For events A and B with P(B) > 0:
```
P(A|B) = P(A ∩ B) / P(B)
```

**Interpretation**: Restrict attention to outcomes in B, renormalize probabilities.

### Chain Rule
For multiple events:
```
P(A ∩ B ∩ C) = P(C) × P(B|C) × P(A|B ∩ C)
```

### Law of Total Probability
If {Bᵢ} partitions Ω:
```
P(A) = Σᵢ P(A|Bᵢ) × P(Bᵢ)
```

### Visual Interpretation
Conditional probability can be visualized using Venn diagrams:
- Event B defines the restricted sample space
- P(A|B) is the fraction of B occupied by A ∩ B
- This corresponds to the area of intersection divided by area of B

### Card Example
Standard deck of 52 cards:
- A = "card is a King" → P(A) = 4/52 = 1/13
- B = "card is a face card" → P(B) = 12/52 = 3/13
- A ∩ B = "King and face card" → P(A ∩ B) = 4/52 = 1/13
- P(A|B) = (1/13) / (3/13) = 1/3

**Intuition**: Among the 12 face cards, 4 are Kings, so 4/12 = 1/3.

## 6. Bayes' Theorem

### Statement
For events A and B with P(B) > 0:
```
P(A|B) = [P(B|A) × P(A)] / P(B)
```

**Components**:
- **Posterior**: P(A|B) - updated belief after seeing evidence
- **Likelihood**: P(B|A) - how well evidence fits hypothesis
- **Prior**: P(A) - initial belief
- **Marginal**: P(B) - total probability of evidence

### Medical Diagnosis Example
Disease screening test:
- P(Disease) = 0.001 (prevalence)
- P(Positive|Disease) = 0.99 (sensitivity)
- P(Positive|No Disease) = 0.01 (false positive rate)

**Calculate P(Disease|Positive)**:
1. P(Positive) = 0.99×0.001 + 0.01×0.999 = 0.01098
2. P(Disease|Positive) = (0.99×0.001) / 0.01098 ≈ 0.0902

**Interpretation**: A positive test result means ~9% chance of actually having the disease.

### Negative Test Result
P(Disease|Negative) = (0.01×0.001) / 0.98902 ≈ 0.00001011

**Interpretation**: A negative test makes disease extremely unlikely (~0.001%).

## 7. Independence

### Definition
Events A and B are independent if:
```
P(A ∩ B) = P(A) × P(B)
```

**Equivalent formulations** (when probabilities > 0):
```
P(A|B) = P(A) and P(B|A) = P(B)
```

### Visual Interpretation
Independent events in Venn diagram:
- Areas don't affect each other
- Intersection area equals product of individual areas
- Knowing one event tells you nothing about the other

### Coin Toss Example
Two fair coins:
- A = "First coin is Heads" → P(A) = 1/2
- B = "Second coin is Heads" → P(B) = 1/2
- A ∩ B = "Both heads" → P(A ∩ B) = 1/4
- Since 1/4 = (1/2)×(1/2), A and B are independent

### Non-Independence Example
- A = "First coin is Heads" → P(A) = 1/2
- D = "At least one Head" → P(D) = 3/4
- A ∩ D = "First coin is Heads" → P(A ∩ D) = 1/2
- Product: (1/2)×(3/4) = 3/8 ≠ 1/2
- Therefore A and D are not independent

### Mutual vs Pairwise Independence

**Pairwise independence**: Every pair of events is independent
**Mutual independence**: Every finite collection of events is independent

**Key insight**: Pairwise independence does NOT imply mutual independence.

### Counterexample
Three events from two coin tosses:
- A = "First coin is Heads"
- B = "Second coin is Heads"
- C = "Coins show same result"

**Pairwise independent**: All pairs satisfy the independence condition
**Not mutually independent**: P(A ∩ B ∩ C) = 1/4 ≠ (1/2)×(1/2)×(1/2) = 1/8

## 8. Summary

This lesson introduced the foundational concepts that underpin all of statistics:

1. **Motivation**: Statistics as the inverse of probability
2. **Applications**: Real-world problems across multiple domains
3. **Probability spaces**: Formal framework for reasoning under uncertainty
4. **Conditional probability**: Updating beliefs with new information
5. **Bayes' theorem**: Systematic way to incorporate evidence
6. **Independence**: When events don't affect each other

These concepts provide the mathematical foundation for all subsequent lessons in the course.

## 9. References and Further Reading

- **Textbook**: "Introduction to Probability and Statistics" by Mendenhall, Beaver, and Beaver
- **Online resources**: Khan Academy Statistics and Probability
- **Advanced**: "Probability and Statistics" by DeGroot and Schervish
- **Applications**: "Bayesian Data Analysis" by Gelman et al.

## 10. Discussion Questions

1. How does the duality between probability and statistics manifest in real-world applications?
2. Why is Bayes' theorem particularly useful in medical diagnosis?
3. What are the practical implications of the distinction between pairwise and mutual independence?
4. How might the Cox-Jaynes theorem influence your approach to data analysis?

---

## Practical Exercises

### Exercise 1: Conditional Probability Practice
Consider a medical test with:
- P(Disease) = 0.01
- P(Positive|Disease) = 0.95
- P(Positive|No Disease) = 0.05

Calculate:
1. P(Disease|Positive)
2. P(Disease|Negative)
3. P(No Disease|Positive)

### Exercise 2: Independence Assessment
For three events A, B, C with:
- P(A) = 0.5, P(B) = 0.3, P(C) = 0.4
- P(A ∩ B) = 0.15, P(A ∩ C) = 0.2, P(B ∩ C) = 0.12
- P(A ∩ B ∩ C) = 0.06

Determine if these events are pairwise independent, mutually independent, or neither.

### Exercise 3: Bayes' Theorem Application
A factory produces widgets with a defect rate of 2%. The quality control system catches 98% of defective widgets and incorrectly rejects 1% of good widgets.

If a widget fails quality control:
1. What is the probability it is actually defective?
2. What is the probability it is actually good?

### Exercise 4: Real-World Independence
Consider the following pairs of events and determine if they are likely to be independent:
1. "It rains today" and "I carry an umbrella"
2. "Stock A increases" and "Stock B increases" (in the same market)
3. "A student studies" and "A student gets good grades"
4. "A coin lands heads" and "The next coin lands heads"

Explain your reasoning for each case.
