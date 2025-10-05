# Mathematical Foundations of NFL Betting Model Validation

## 1. Time-Series Cross-Validation

### Walk-Forward Validation
Walk-forward validation respects temporal order and prevents data leakage:

**Mathematical Framework:**
- Training window: `W_train = [t - 3*seasons, t]`
- Purge gap: `G_purge = [t, t + 1 week]`
- Test window: `W_test = [t + 1 week, t + 5 weeks]`
- Embargo: `G_embargo = [t + 5 weeks, t + 7 weeks]`

**Why 3-season training windows?**
- Captures seasonal patterns
- Sufficient data for stable estimates
- Balances recency vs. sample size

**Temporal gaps prevent:**
1. **Look-ahead bias**: Using future information
2. **Data leakage**: Correlated observations between train/test
3. **Overfitting**: Model memorizing recent patterns

### Mathematical Justification for Gaps

**Autocorrelation decay:**
```
ρ(k) = Cov(X_t, X_{t+k}) / Var(X_t)
```

For NFL games, autocorrelation typically decays to insignificance after 1-2 weeks.

**Purge period calculation:**
- Minimum gap = max(autocorrelation_lag) + 1
- For NFL: 1 week purge ensures independence

**Embargo period:**
- Prevents using test information in next training
- 2-week embargo accounts for team adjustments

---

## 2. Statistical Significance Testing

### Permutation Test
Tests null hypothesis H₀: Model has no predictive power

**Algorithm:**
```python
observed_metric = f(y_true, y_pred)
for i in 1:N_permutations:
    y_pred_shuffled = random_permutation(y_pred)
    null_metrics[i] = f(y_true, y_pred_shuffled)
p_value = P(null_metrics >= observed_metric)
```

**Mathematical basis:**
- Under H₀, prediction order is exchangeable
- Permutation distribution approximates null distribution
- One-sided test for betting (we want P > breakeven)

**Effect size (Cohen's d):**
```
d = (μ_observed - μ_null) / σ_null
```
- d > 0.8: Large effect
- d > 0.5: Medium effect
- d > 0.2: Small effect

### Bootstrap Confidence Intervals
Non-parametric confidence intervals for win rate:

**Bootstrap principle:**
```
F̂_n → F as n → ∞
```
Where F̂_n is empirical distribution

**Percentile method:**
```
CI = [B^(-1)(α/2), B^(-1)(1-α/2)]
```
Where B^(-1) is inverse of bootstrap distribution

**Bias-corrected accelerated (BCa) intervals:**
More accurate for skewed distributions:
```
CI = [B^(-1)(Φ(z₀ + (z₀ + z_α)/(1 - a(z₀ + z_α)))),
      B^(-1)(Φ(z₀ + (z₀ + z_{1-α})/(1 - a(z₀ + z_{1-α})))]
```

### Minimum Sample Size Calculation
For detecting edge in binary outcomes:

**Power analysis for proportions:**
```
n = (Z_α + Z_β)² × [p₁(1-p₁) + p₀(1-p₀)] / (p₁ - p₀)²
```

Where:
- p₀ = 0.5238 (breakeven at -110 odds)
- p₁ = p₀ + effect_size
- Z_α = 1.96 (95% confidence)
- Z_β = 0.84 (80% power)

**Example:** For 3% edge:
```
n = (1.96 + 0.84)² × [0.5538×0.4462 + 0.5238×0.4762] / (0.03)²
n ≈ 1,367 games
```

---

## 3. Regime Change Detection

### CUSUM (Cumulative Sum Control Chart)
Detects shifts in process mean:

**CUSUM statistics:**
```
S⁺ₜ = max(0, S⁺ₜ₋₁ + (xₜ - μ₀ - K))
S⁻ₜ = min(0, S⁻ₜ₋₁ + (xₜ - μ₀ + K))
```

Where:
- μ₀ = target value (expected return)
- K = reference value (typically 0.5σ)
- Signal when |S| > h (threshold)

**Threshold selection:**
```
h = 3σ (for 99.7% confidence)
```

**Average Run Length (ARL):**
```
ARL₀ = exp(2μh/σ²) (in-control)
ARL₁ = h/δ (out-of-control)
```

### Regime Performance Testing
Test model across known structural breaks:

**Chow test for structural break:**
```
F = [(RSS_pooled - RSS₁ - RSS₂)/k] / [(RSS₁ + RSS₂)/(n₁ + n₂ - 2k)]
```

Under H₀: F ~ F(k, n₁ + n₂ - 2k)

### Feature Stability Tracking

**Coefficient of Variation:**
```
CV = σ/μ
```
- CV < 0.3: Stable feature
- CV > 0.5: Unstable feature

**Rank correlation over time:**
```
ρ_rank = 1 - (6Σd²ᵢ)/(n(n²-1))
```
Where dᵢ = rank difference at time i

---

## 4. Betting-Specific Metrics

### Sharpe Ratio (Annualized)
Risk-adjusted returns:

```
Sharpe = (E[R] - Rₑ) / σ(R) × √(periods_per_year)
```

For NFL (267 games/season):
```
Sharpe_annual = Sharpe_game × √267
```

**Interpretation:**
- Sharpe > 2.0: Excellent
- Sharpe > 1.0: Good
- Sharpe > 0.5: Acceptable
- Sharpe < 0.5: Poor

### Maximum Drawdown
Largest peak-to-trough decline:

```
DD(t) = max(0, max(P(s)) - P(t)) for s in [0,t]
MDD = max(DD(t)) for all t
```

**Recovery metrics:**
- Drawdown duration: t_end - t_start
- Recovery time: t_recovery - t_end
- Underwater curve: Current DD at each point

### Closing Line Value (CLV)
Measures if beating the closing line:

```
CLV = E_model - E_closing
```

Where E = implied edge from probability

**CLV correlation with profit:**
```
ρ(CLV, Profit) > 0.3 indicates predictive CLV
```

### Kelly Criterion Optimization
Optimal bet sizing for geometric growth:

**Full Kelly:**
```
f* = (p×b - q) / b
```
Where:
- p = win probability
- q = 1 - p
- b = net odds (decimal - 1)

**Expected growth rate:**
```
g = p×ln(1 + b×f) + q×ln(1 - f)
```

**Fractional Kelly (conservative):**
```
f = k × f* where k ∈ [0.2, 0.3]
```

**Why fractional Kelly?**
1. Parameter uncertainty
2. Non-normal returns
3. Psychological comfort

---

## 5. Multiple Testing Correction

### Benjamini-Hochberg FDR Control
Controls False Discovery Rate:

**Algorithm:**
1. Order p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎
2. Find largest i: p₍ᵢ₎ ≤ (i/m) × α
3. Reject H₍₁₎, ..., H₍ᵢ₎

**FDR calculation:**
```
FDR = E[V/R | R > 0] × P(R > 0)
```
Where V = false positives, R = total positives

**Why FDR over FWER?**
- Less conservative than Bonferroni
- Better power for many hypotheses
- Appropriate for exploratory analysis

---

## 6. Calibration and Reliability

### Expected Calibration Error (ECE)
Measures probability calibration:

```
ECE = Σᵢ (nᵢ/n) × |acc(Bᵢ) - conf(Bᵢ)|
```

Where for bin Bᵢ:
- acc = actual frequency
- conf = mean predicted probability
- nᵢ = samples in bin

### Brier Score Decomposition
```
BS = Reliability - Resolution + Uncertainty
```

Where:
- Reliability: Calibration quality
- Resolution: Ability to discriminate
- Uncertainty: Inherent randomness

### Isotonic Regression Calibration
Monotonic transformation for calibration:

```
min Σᵢ wᵢ(yᵢ - ŷᵢ)²
subject to: ŷᵢ ≤ ŷⱼ for i < j
```

---

## 7. Risk Metrics

### Value at Risk (VaR)
Maximum loss at confidence level:

```
VaR_α = -inf{x : P(R ≤ x) ≥ α}
```

For normal returns:
```
VaR_α = μ - z_α × σ
```

### Conditional Value at Risk (CVaR)
Expected loss beyond VaR:

```
CVaR_α = E[R | R ≤ VaR_α]
```

**Coherent risk measure properties:**
1. Monotonicity
2. Sub-additivity
3. Positive homogeneity
4. Translation invariance

### Sortino Ratio
Downside risk-adjusted returns:

```
Sortino = (E[R] - MAR) / σ_downside
```

Where:
```
σ_downside = √(E[min(R - MAR, 0)²])
```

---

## 8. Implementation Considerations

### Computational Complexity
- Walk-forward validation: O(n × k × model_complexity)
- Permutation test: O(n × n_permutations)
- Bootstrap: O(n × n_bootstrap)
- CUSUM: O(n)

### Numerical Stability
- Use log-space for probabilities
- Standardize features before VIF
- Handle division by zero in ratios

### Statistical Power
Required samples for various tests:
- Permutation test: n > 30
- Bootstrap CI: n > 100
- Regime detection: n > 500 per regime
- Feature stability: n > 1000

---

## References

1. Efron, B., & Tibshirani, R. (1993). An Introduction to the Bootstrap.
2. López de Prado, M. (2018). Advances in Financial Machine Learning.
3. Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.
4. Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate.
5. Page, E. S. (1954). Continuous Inspection Schemes.
6. Kelly, J. L. (1956). A New Interpretation of Information Rate.
