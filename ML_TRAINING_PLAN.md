# ML Training Pipeline: Failure Learning, Diversity & Exploration Plan

## Problem Statement

The current ML pipeline has three fundamental flaws:

1. **No Failure Learning**: Failed builds are discarded. ML only sees passing builds → can't learn what makes a build bad
2. **No Diversity**: Optimizing for single objective → mode collapse to one build archetype  
3. **No Exploration**: Random generation + filter doesn't explore the vast possibility space intelligently

## Current Architecture (Broken)

```
Generate → Precheck → PoB Evaluate → Gate Check → Keep if PASS, Discard if FAIL
                                                          ↑
                                          ML ONLY SEES PASSES!
```

## Proposed Architecture

```
Generate → Precheck → PoB Evaluate → Gate Check → Store ALL (pass OR fail)
                                                          ↓
                                              ML trains on PASS + FAIL
                                                          ↓
                                              Diversity Sampling
                                                          ↓
                                              Exploration Strategy
                                                          ↓
                                              Smart Candidate Selection
```

---

## Issue 1: Failure Learning

### Problem
- Failed builds cleaned up: `_cleanup_candidate()` deletes DB record + artifacts
- `persist=false` records not in `generation_records`
- ML dataset only contains successful builds

### Solution: Store ALL Results

**Changes needed:**

1. **Keep failed build artifacts** (don't delete)
   - Modify `runner.py`: `_cleanup_candidate` should NOT run on failure
   - Keep artifacts for analysis/debugging

2. **Add `gate_pass` column to ALL stored records**
   - `generation_records` already has `gate_pass` field
   - Just need to ensure failures ARE stored (not filtered out)

3. **Include failure reasons in training data**
   - `gate_fail_reasons` already captured
   - Include as features in ML dataset

4. **Train on positive AND negative examples**
   - Binary classification: pass vs fail
   - Or regression with penalty for failure reasons
   - Contrastive loss: learn "distance" from passing builds

---

## Issue 2: Diversity (Avoid Mode Collapse)

### Problem
- Current: keep top-N by score → converges to single archetype
- No clustering or archetype awareness
- Phase 5 (Quality-Diversity archive) not implemented

### Solution: MAP-Elites / Clustering

**Approach 1: MAP-Elites (Quality-Diversity)**

```
1. Define niche dimensions:
   - Class (6 options)
   - Damage type (physical/chaos/elemental/hybrid)
   - Defense type (armor/evasion/energy shield/hybrid)
   - Main skill (varies)
   
2. For each iteration:
   a. Generate candidates
   b. Evaluate in PoB
   c. Place each in appropriate niche
   d. Keep best from each niche
   e. If niche empty → randomly explore
   f. If niche full but new better → replace
   
3. Result: diverse archive with best from each niche
```

**Approach 2: Diversity-Aware Selection**

```
1. Cluster existing verified builds
2. When selecting for next iteration:
   - 70% from best overall (exploitation)
   - 30% from underrepresented clusters (exploration)
3. Penalize similar builds in selection
```

**Implementation:**

1. Add diversity clustering in runner
2. Store archetype tags per build
3. Selection algorithm with diversity bonus

---

## Issue 3: Exploration Strategy

### Problem
- Vast space: ~500 passive nodes × skill gems × items
- Random sampling is inefficient
- No intelligent exploration

### Solution: Multi-Armed Bandit / Novelty Search

**Epsilon-Greedy:**
```
- ε = 0.1 → 10% random, 90% exploitation
- Decay ε over time as model improves
```

**Novelty Search:**
```
- Build "novelty score" based on:
  - Unused passive tree regions
  - Rare gem combinations
  - Unique item choices
- Reward novel builds even if low score
```

**Curiosity-Driven:**
```
- Track what model learns poorly
- Generate candidates targeting weak areas
- Maximize learning progress
```

**Multi-Objective Pareto:**
```
- Optimize for: DPS, Defense, Cost, Survivability
- Keep Pareto frontier builds
- Explore tradeoff space
```

---

## Implementation Roadmap

### Phase 1: Failure Tracking (Quick Win)
- [ ] Remove `_cleanup_candidate` on failure OR make optional
- [ ] Ensure all evaluation records stored in `generation_records`
- [ ] Verify `gate_pass` is captured for all
- [ ] Update ML dataset builder to include failures

### Phase 2: Diversity Sampling
- [ ] Define niche dimensions for PoE builds
- [ ] Implement clustering on existing verified builds
- [ ] Add diversity-aware selection to optimizer
- [ ] Track diversity metrics per iteration

### Phase 3: Exploration Strategy  
- [ ] Implement ε-greedy selection (configurable ε)
- [ ] Add novelty scoring based on genome features
- [ ] Add curiosity-driven target selection
- [ ] Tune exploration vs exploitation balance

### Phase 4: Full Integration
- [ ] Combine: failures + diversity + exploration
- [ ] A/B test different strategies
- [ ] Measure diversity improvement
- [ ] Validate ML improvements

---

## Key Files to Modify

1. **runner.py** - `run_generation()` 
   - Don't cleanup failed candidates
   - Store all evaluation results
   
2. **ml_loop.py** - `build_dataset_snapshot()`
   - Include failure examples
   - Add diversity sampling

3. **surrogate.py** - `train()`
   - Support binary classification (pass/fail)
   - Or contrastive loss

4. **New: diversity.py** - `select_diverse_candidates()`
   - Clustering logic
   - Niche selection

5. **New: exploration.py** - `explore()`
   - Epsilon-greedy
   - Novelty search
   - Curiosity-driven

---

## Acceptance Criteria

### Phase 1 (Failure Learning):
- [ ] All builds (pass OR fail) stored in ClickHouse
- [ ] `gate_pass` column populated for all
- [ ] ML dataset includes both pass and fail examples
- [ ] ML can predict failure reasons

### Phase 2 (Diversity):
- [ ] At least 5 distinct archetypes in archive
- [ ] Each archetype has elite builds
- [ ] Selection favors underrepresented archetypes

### Phase 3 (Exploration):
- [ ] Configurable exploration rate
- [ ] Novel builds discovered beyond initial population
- [ ] Exploration decreases as model improves

### Phase 4 (Integration):
- [ ] ML improves over iterations (not just random)
- [ ] Archive contains diverse, high-quality builds
- [ ] System explores novel build types
