# Feature Validation Strategy: What Actually Predicts NFL Games?
**Goal:** Systematically determine which features predict game outcomes
**Method:** Evidence-based validation using your existing framework

---

## The Core Question

**What data actually predicts NFL game outcomes?**

Not opinions. Not hunches. **Empirical validation.**

---

## Part 1: The Academic Foundation (What Sharp Bettors Use)

### Tier 1: Proven Predictive Features (Use These)

Based on academic research and professional sports betting:

**1. EPA (Expected Points Added)** - HIGHEST PREDICTIVE POWER
- Offensive EPA: ~0.15 correlation with wins
- Defensive EPA: ~0.12 correlation with wins
- EPA differential: ~0.22 correlation with point spread
- **Your system already has this** ✅

**2. Recent Performance (Last 3-5 Games)**
- Win rate last 3 games: ~0.18 correlation
- Scoring differential trend: ~0.16 correlation
- EPA trend: ~0.14 correlation
- **Exponential decay** matters (recent > old)

**3. Rest Advantage**
- Extra rest day advantage: ~2 points
- Thursday games: Home team -1.5 points
- Traveling west-to-east: -0.5 points
- **Easy to calculate** from schedules

**4. Home Field Advantage**
- Average: ~2.5 points (but declining over time)
- Varies by stadium: 1.5-4 points
- Dome teams outdoors: -1 point
- **Simple binary feature**

**5. Scoring Efficiency**
- Red zone TD%: ~0.13 correlation
- Third down conversion: ~0.11 correlation
- Turnover differential: ~0.19 correlation
- **All in play-by-play data**

### Tier 2: Useful But Weaker (Add If Tier 1 Works)

**6. Strength of Schedule**
- Opponent win rate: ~0.08 correlation
- Opponent EPA: ~0.07 correlation
- Worth including

**7. Divisional Games**
- Familiarity reduces home advantage: -0.5 points
- Historical matchups matter: ~0.06 correlation
- Easy binary feature

**8. Weather (Outdoor Games Only)**
- Wind >15mph: Affects totals (~3 points lower)
- Temperature <32°F: Affects totals (~2 points lower)
- Rain/Snow: Minimal impact (surprising!)
- **Only matters for totals, not spreads**

### Tier 3: Overhyped (Might Not Help)

**9. Injuries**
- QB injury: HUGE (~7 point swing)
- Other positions: Marginal (~0.5-1 point)
- **Problem:** Historical injury data is sparse
- **Verdict:** Use for QB only if available

**10. Coaching Changes**
- New coach bump: ~0.5 points (overrated)
- Only first 3 games
- Minimal long-term impact

**11. Motivational Factors**
- "Trap games", "look ahead" → NOT predictive
- "Prime time" advantage → Minimal
- "Revenge game" → No correlation
- **Ignore these**

---

## Part 2: Your Existing Validation Framework

**Good news:** You already built this! `validation/data_validation_framework.py`

### Use Your 5-Phase Framework:

**Phase 1: Statistical Foundation**
- Test each feature for correlation with outcomes
- Calculate p-values
- Identify multicollinearity
- **File:** `validation/data_validation_framework.py:163`

**Phase 2: Market Validation**
- Do your features predict better than betting lines?
- Test CLV (Closing Line Value)
- **File:** `validation/market_efficiency_tester.py:230`

**Phase 3: Temporal Stability**
- Do features work across different seasons?
- Check for regime changes
- **File:** `validation/temporal_stability_analyzer.py`

**Phase 4: Implementation Strategy**
- Feature importance from XGBoost
- SHAP values
- Ablation testing

**Phase 5: Performance Monitoring**
- Track feature drift over time
- Re-validate quarterly

---

## Part 3: Systematic Feature Validation Process

### Step 1: Calculate Feature Universe (2 hours)

For each game in your 2,700-game dataset, calculate:

```python
# Per-game features
features = {
    # EPA-based (Tier 1)
    'home_off_epa_last5': [...],
    'home_def_epa_last5': [...],
    'away_off_epa_last5': [...],
    'away_def_epa_last5': [...],
    'epa_differential': [...],

    # Scoring efficiency (Tier 1)
    'home_redzone_td_pct': [...],
    'away_redzone_td_pct': [...],
    'home_3rd_down_pct': [...],
    'away_3rd_down_pct': [...],
    'home_turnover_diff': [...],
    'away_turnover_diff': [...],

    # Situational (Tier 1)
    'rest_advantage_days': [...],
    'is_home': [...],
    'is_dome': [...],
    'week_number': [...],
    'is_divisional': [...],

    # Recent form (Tier 1)
    'home_win_pct_last3': [...],
    'away_win_pct_last3': [...],
    'home_scoring_diff_last3': [...],
    'away_scoring_diff_last3': [...],

    # Weather (Tier 2 - for totals)
    'temperature': [...],
    'wind_speed': [...],

    # Strength of schedule (Tier 2)
    'home_opp_win_pct': [...],
    'away_opp_win_pct': [...],
}
```

**Total features:** ~25-30 to start

### Step 2: Univariate Correlation Analysis (1 hour)

Test each feature individually:

```python
# test_feature_correlations.py

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def test_single_feature(feature_name, feature_values, targets):
    """Test correlation of single feature with outcomes"""

    # Test 1: Correlation with point differential
    corr_diff, p_value_diff = pearsonr(feature_values, targets['point_diff'])

    # Test 2: Correlation with home cover
    corr_cover, p_value_cover = pearsonr(feature_values, targets['home_cover'])

    return {
        'feature': feature_name,
        'corr_point_diff': corr_diff,
        'p_value_diff': p_value_diff,
        'corr_home_cover': corr_cover,
        'p_value_cover': p_value_cover,
        'predictive_power': abs(corr_diff)  # Absolute correlation
    }

# Run for all features
results = []
for feature in features:
    result = test_single_feature(feature, X[feature], y)
    results.append(result)

# Sort by predictive power
results_df = pd.DataFrame(results)
results_df.sort_values('predictive_power', ascending=False, inplace=True)

print(results_df.head(15))
```

**Expected Output:**
```
Feature                    | Corr  | P-value | Predictive Power
---------------------------|-------|---------|------------------
epa_differential          | 0.22  | <0.001  | 0.22  ⭐⭐⭐
home_turnover_diff        | 0.19  | <0.001  | 0.19  ⭐⭐⭐
recent_win_pct_diff       | 0.18  | <0.001  | 0.18  ⭐⭐
rest_advantage            | 0.14  | <0.001  | 0.14  ⭐⭐
home_redzone_td_pct       | 0.13  | <0.001  | 0.13  ⭐⭐
...
```

**Decision Rules:**
- **Keep:** p-value < 0.01 AND |correlation| > 0.10
- **Maybe:** p-value < 0.05 AND |correlation| > 0.05
- **Drop:** Everything else

### Step 3: Multivariate Feature Importance (2 hours)

Use XGBoost to test feature combinations:

```python
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Train model on ALL features
model = XGBRegressor(n_estimators=100, max_depth=4)
model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

# Plot top 20
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['feature'][:20],
         feature_importance_df['importance'][:20])
plt.title('Top 20 Most Important Features')
plt.savefig('feature_importance.png')

print(feature_importance_df.head(20))
```

**Decision Rules:**
- **Core features:** Top 10 by importance (>5%)
- **Supplementary:** Next 10 (2-5%)
- **Drop:** Everything else (<2%)

### Step 4: Ablation Testing (3 hours)

Test what happens when you remove features:

```python
def ablation_test(X, y, feature_to_remove):
    """Test model performance without a feature"""

    # Baseline: All features
    X_all = X.copy()
    baseline_score = cross_val_score(model, X_all, y, cv=5).mean()

    # Test: Remove one feature
    X_ablated = X.drop(columns=[feature_to_remove])
    ablated_score = cross_val_score(model, X_ablated, y, cv=5).mean()

    # Impact of removing
    impact = baseline_score - ablated_score

    return {
        'feature': feature_to_remove,
        'baseline': baseline_score,
        'without': ablated_score,
        'impact': impact
    }

# Test removing each feature
ablation_results = []
for feature in X.columns:
    result = ablation_test(X, y, feature)
    ablation_results.append(result)

# Sort by impact
ablation_df = pd.DataFrame(ablation_results)
ablation_df.sort_values('impact', ascending=False, inplace=True)

print(ablation_df.head(15))
```

**Decision Rules:**
- **Critical:** Impact > 0.02 (2% accuracy drop)
- **Important:** Impact > 0.01 (1% accuracy drop)
- **Marginal:** Impact > 0.005 (0.5% accuracy drop)
- **Useless:** Impact ≤ 0.005 (remove)

### Step 5: Temporal Validation (2 hours)

Ensure features work across time:

```python
# Test features by season
seasons = X['season'].unique()

temporal_results = []
for season in seasons:
    # Train on all OTHER seasons
    train_mask = X['season'] != season
    test_mask = X['season'] == season

    model.fit(X[train_mask], y[train_mask])
    score = model.score(X[test_mask], y[test_mask])

    temporal_results.append({
        'test_season': season,
        'accuracy': score
    })

temporal_df = pd.DataFrame(temporal_results)
print(temporal_df)

# Check for drift
std_dev = temporal_df['accuracy'].std()
if std_dev > 0.05:
    print("⚠️  HIGH VARIANCE across seasons - features may be unstable")
else:
    print("✅ Features are temporally stable")
```

---

## Part 4: My Predictions (What Will Work)

Based on NFL betting research, here's what I expect:

### Features That Will Be Top 5:
1. ✅ **EPA differential** (off_epa - def_epa for each team)
2. ✅ **Recent performance** (last 3-5 games scoring diff)
3. ✅ **Rest advantage** (days between games)
4. ✅ **Turnover differential** (from play-by-play)
5. ✅ **Home field advantage** (binary indicator)

### Features That Will Be Useful (Top 6-15):
6. ✅ Red zone efficiency
7. ✅ Third down conversion rate
8. ✅ Divisional game indicator
9. ✅ Strength of schedule
10. ✅ Win percentage last 3 games
11. ✅ Week number (early vs late season)
12. ✅ Scoring efficiency
13. ✅ Pass vs rush EPA
14. ✅ Time of possession
15. ✅ Explosive play rate (20+ yard gains)

### Features That Probably Won't Help:
- ❌ Coaching tenure
- ❌ "Momentum" indicators
- ❌ Injuries (except QB)
- ❌ Travel distance
- ❌ Prime time game indicator
- ❌ Revenge game narratives

### For Totals (Over/Under) Specifically:
- ✅ Combined offensive EPA (strong predictor)
- ✅ Combined pace of play
- ✅ Wind speed >15mph (lowers totals)
- ✅ Temperature <32°F (lowers totals)
- ✅ Dome game indicator

---

## Part 5: The Validation Timeline

### Week 1: Feature Calculation (8 hours)
- Extract features from 2,700 games
- Calculate all Tier 1 + Tier 2 features
- Save to feature_matrix.pkl

### Week 2: Validation Testing (10 hours)
- Univariate correlations (1 hour)
- Multivariate importance (2 hours)
- Ablation testing (3 hours)
- Temporal validation (2 hours)
- Document results (2 hours)

### Week 3: Feature Selection (4 hours)
- Drop useless features
- Keep top 15-20
- Re-test model performance
- Finalize feature set

**Total:** 22 hours to scientifically validate features

---

## Part 6: Expected Outcomes

### Conservative Estimate:
- **10-12 features** will be truly predictive
- **5-8 features** will be marginally useful
- **10+ features** will be noise

### Realistic Feature Set:
```python
final_features = [
    # Core predictors (80% of signal)
    'epa_differential',
    'recent_performance_diff',
    'rest_advantage',
    'home_indicator',
    'turnover_diff',

    # Supplementary (15% of signal)
    'redzone_efficiency_diff',
    'third_down_diff',
    'strength_of_schedule',
    'divisional_game',
    'week_number',

    # Nice-to-have (5% of signal)
    'explosive_play_rate',
    'time_of_possession',
    'scoring_efficiency',
]
```

**With 13 features and 2,700 games:**
- 207 games per feature (well above minimum 100)
- Low overfitting risk
- Expected accuracy: **52-54%**

---

## Part 7: The Ultimate Test (Your Secret Weapon)

### Use Your Validation Framework!

You already built `validation/market_efficiency_tester.py` - USE IT!

**Test:**
1. Do your features beat market closing lines?
2. Do they generate positive CLV?
3. Do they predict better than simple metrics?

```python
# validation/test_our_features.py

from validation.market_efficiency_tester import MarketEfficiencyTester

tester = MarketEfficiencyTester()

# Test Tier 1 features
tier1_features = ['epa_diff', 'rest', 'home', 'turnover_diff']
tier1_results = tester.test_feature_set(X[tier1_features], y)

# Test Tier 1 + Tier 2
all_features = tier1_features + ['redzone_pct', 'third_down', 'sos']
all_results = tester.test_feature_set(X[all_features], y)

# Compare
print(f"Tier 1 only: {tier1_results['clv']:.2%} CLV")
print(f"All features: {all_results['clv']:.2%} CLV")

if all_results['clv'] > tier1_results['clv'] + 0.01:
    print("✅ Tier 2 features add value")
else:
    print("❌ Tier 2 features don't help, stick with Tier 1")
```

---

## Part 8: Quick Start This Week

### Day 1: Feature Calculation
```bash
python calculate_all_features.py --seasons 2015-2024
# Output: feature_matrix.pkl (2,700 games x 30 features)
```

### Day 2: Quick Validation
```bash
python test_feature_correlations.py
python test_feature_importance.py
# Output: Which features matter
```

### Day 3: Train Baseline Model
```bash
python train_with_top10_features.py
# Output: Does 52%+ accuracy look achievable?
```

**Decision Point:**
- If yes → proceed to full training
- If no → need more data or better features

---

## Bottom Line: What's Predictive?

**Proven by research:**
1. EPA differential → #1 predictor
2. Recent performance → Strong
3. Rest advantage → Meaningful
4. Home field → Consistent
5. Turnovers → Important

**Your advantage:**
- You have the validation framework
- You have 2,700+ games
- You have all the right features
- You just need to calculate and test them

**Timeline:** 22 hours to validate features scientifically

**Expected result:** 52-54% accuracy with proper feature selection

**Ready to build the feature validation pipeline?**
