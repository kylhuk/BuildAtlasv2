# ML Training: Failure Learning, Diversity & Exploration - Implementation Plan

**Version:** 1.0
**Created:** 2026-03-01
**Status:** Ready for Implementation

---

## Executive Summary

The current ML pipeline has three critical flaws that prevent effective learning:

| Issue | Current Behavior | Impact |
|-------|------------------|--------|
| **No Failure Learning** | Failed builds purged from DB + disk | ML only sees passing builds → can't learn what makes builds fail |
| **No Diversity** | Selection keeps top-N by score | Mode collapse to single archetype |
| **No Exploration** | Random generation + filter | Can't explore vast possibility space |

This plan addresses all three systematically.

---

# Part 1: Failure Learning (Immediate Priority)

## 1.1 Problem Analysis

### Current Flow (Broken)
```
Generate → Persist Artifacts → PoB Evaluate → Gate Check
                                                  ↓
                              PASS: Keep artifacts + DB record
                              FAIL: DELETE artifacts + DELETE DB record
                                                         ↓
                                              ML NEVER SEES FAILURES!
```

### Root Cause
In `runner.py` lines 1446 and 1469:
```python
if status is not BuildStatus.evaluated:
    evaluation_failures += 1
    _cleanup_candidate(candidate)  # DELETES EVERYTHING!
```

### Evidence
- `_cleanup_candidate()` purges DB and deletes artifacts
- Failed builds have no metrics in `data/builds/`
- `build_dataset_snapshot()` only reads successful builds
- ML dataset contains ONLY passing builds

---

## 1.2 Solution: Store All Results

### Change 1: Don't Delete Failed Artifacts

**File:** `backend/engine/generation/runner.py`

**Current (line 1446):**
```python
else:
    evaluation_failures += 1
    _cleanup_candidate(candidate)
```

**Proposed:**
```python
else:
    evaluation_failures += 1
    # DO NOT DELETE - keep for ML training
    # Failed builds are valuable training examples!
    candidate.evaluation_status = BuildStatus.failed.value
```

**Same change at line 1469 (exception handler)**

### Change 2: Mark Failed Status in DB

**File:** `backend/engine/generation/runner.py`

Current: When `_persist_candidate` is called, it sets status based on `candidate.failures`:
```python
status_value = (
    BuildStatus.failed.value if candidate.failures else BuildStatus.imported.value
)
```

This is BEFORE evaluation. Need to update AFTER evaluation if failed.

**Add after evaluation loop:**
```python
# Update status for failed evaluations
for candidate in candidates:
    if candidate.selected_for_evaluation and candidate.evaluation_status:
        # Status already set by evaluation loop
        # Just ensure DB reflects the final status
        pass  # Already persisted as 'imported', need to update to 'failed'
```

Actually, simpler: don't call `_cleanup_candidate` and update the status in DB.

### Change 3: Include Gate Pass/Fail in Metrics

**File:** `backend/engine/surrogate/dataset.py`

Current `_build_row()` doesn't include gate information. Add:

```python
def _build_row(
    build_id: str,
    scenario_id: str,
    genome: Mapping[str, Any],
    scenario_data: Mapping[str, Any],
    build_details: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    # ... existing code ...
    
    # ADD: Gate pass/fail information
    row["gate_pass"] = scenario_data.get("gate_pass", None)
    row["gate_fail_reasons"] = scenario_data.get("gate_fail_reasons", [])
    
    return row
```

### Change 4: ML Training on Failures

**File:** `backend/engine/surrogate/model.py` - `train()` function

Current: Only trains on successful builds.

**Add binary classification:**
```python
def train(
    dataset_path: Path,
    model_path: Path,
    *,
    include_failures: bool = True,  # NEW PARAMETER
) -> TrainResult:
    """Train surrogate model.
    
    Args:
        dataset_path: Path to training data
        model_path: Where to save model
        include_failures: If True, train on both pass and fail examples
                         for binary classification
    """
    # ... existing code ...
    
    # NEW: Handle failure examples
    if include_failures:
        df["target_pass"] = df["gate_pass"].map(
            lambda x: 1 if x is True else (0 if x is False else None)
        )
        df = df.dropna(subset=["target_pass"])
        
        # Train classifier: will this build pass gates?
        clf = RandomForestClassifier(...)
        clf.fit(df[feature_cols], df["target_pass"])
        
        # Also train regressor on passing builds only
        passing_df = df[df["target_pass"] == 1]
        reg = GradientBoostingRegressor(...)
        reg.fit(passing_df[feature_cols], passing_df["full_dps"])
        
        return TrainResult(
            classifier=clf,
            regressor=reg,
            ...
        )
```

---

## 1.3 Implementation Tasks

| Task ID | Description | File | Lines | Effort |
|---------|------------|------|-------|--------|
| FL-01 | Remove `_cleanup_candidate` calls on evaluation failure | runner.py | 1446, 1469 | 1 hr |
| FL-02 | Update DB status for failed builds | runner.py | ~1500 | 1 hr |
| FL-03 | Add gate_pass to dataset rows | dataset.py | _build_row() | 2 hr |
| FL-04 | Add include_failures param to train() | model.py | train() | 2 hr |
| FL-05 | Test: Verify failed builds persist | - | - | 1 hr |
| FL-06 | Test: Verify ML dataset includes failures | - | - | 1 hr |

---

# Part 2: Diversity (MAP-Elites)

## 2.1 Problem Analysis

### Current Selection (Broken)
```python
# Select top-N by score
elites = sorted(candidates, key=lambda c: c.score, reverse=True)[:elite_count]
```

This always picks the same archetype → mode collapse.

### What We Need
```
Archive should contain:
- Best physical melee build
- Best chaos spell build  
- Best elemental totem build
- Best minion build
- Best CI build
- Best life-based build
... and so on
```

---

## 2.2 Solution: MAP-Elites / Quality-Diversity

### Concept
Define "niche" dimensions, keep best from each niche:

```
                Defense
                   ↑
   Armor          |           Evasion
      ●           |              ●
      ●●●●●●●●●●●●●┼●●●●●●●●●●●●●
      ●           |           ●●●
      ●           |              ●
                   |
                   +--------------------→ Offense
                 Low DPS              High DPS
```

### Niche Dimensions for PoE

```python
NICHE_DIMENSIONS = {
    "class": ["Marauder", "Ranger", "Witch", "Templar", "Shadow", "Scion"],
    "damage_type": ["physical", "chaos", "elemental", "hybrid"],
    "defense_type": ["armor", "evasion", "energy_shield", "hybrid"],
    "main_skill_type": ["spell", "attack", "minion", " totem", "trap"],
}
```

### Implementation: Diversity Selection

**New File:** `backend/engine/generation/diversity.py`

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class NicheAssignment:
    class_name: str
    damage_type: str  
    defense_type: str
    skill_type: str

def assign_niche(build: dict[str, Any]) -> NicheAssignment:
    """Assign a build to a niche based on its characteristics."""
    return NicheAssignment(
        class_name=build.get("class", "unknown"),
        damage_type=_infer_damage_type(build),
        defense_type=build.get("defense_archetype", "hybrid"),
        skill_type=_infer_skill_type(build),
    )

def select_diverse_elites(
    candidates: list[Candidate],
    elite_count: int,
    archive: dict[NicheAssignment, Candidate] | None = None,
) -> list[tuple[Candidate, float]]:
    """Select elites that cover diverse niches.
    
    Args:
        candidates: Available candidates
        elite_count: How many elites to select
        archive: Existing archive of best per niche
        
    Returns:
        List of (candidate, score) tuples, diversified across niches
    """
    if not archive:
        archive = {}
    
    # Assign each candidate to a niche
    niche_scores: dict[NicheAssignment, list[tuple[Candidate, float]]] = {}
    
    for candidate in candidates:
        niche = assign_niche(candidate.to_dict())
        if niche not in niche_scores:
            niche_scores[niche] = []
        niche_scores[niche].append((candidate, candidate.score))
    
    # Select best from each niche
    selected = []
    for niche, scored_candidates in niche_scores.items():
        if scored_candidates:
            best = max(scored_candidates, key=lambda x: x[1])
            selected.append(best)
    
    # If we need more, fill with highest scoring remaining
    selected_scores = set(id(c) for c, _ in selected)
    remaining = [(c, s) for c, s in candidates 
                 if id(c) not in selected_scores]
    remaining.sort(key=lambda x: x[1], reverse=True)
    
    while len(selected) < elite_count and remaining:
        selected.append(remaining.pop(0))
    
    return selected[:elite_count]
```

---

## 2.3 Archive Management

**New File:** `backend/engine/archive/diversity_archive.py`

```python
class DiversityArchive:
    """MAP-Elites style archive maintaining best per niche."""
    
    def __init__(self, niche_dimensions: dict[str, list[str]]):
        self.niche_dims = niche_dimensions
        self.archive: dict[NicheAssignment, dict[str, Any]] = {}
    
    def update(self, build: dict[str, Any]) -> bool:
        """Add build to archive if it improves its niche.
        
        Returns:
            True if build was added to archive
        """
        niche = assign_niche(build)
        score = build.get("full_dps", 0)
        
        existing = self.archive.get(niche)
        if existing is None or score > existing.get("full_dps", 0):
            self.archive[niche] = build
            return True
        return False
    
    def get_diverse_sample(self, n: int) -> list[dict]:
        """Get n builds, one per niche if possible."""
        niches = list(self.archive.keys())
        if not niches:
            return []
        
        # Shuffle for variety
        random.shuffle(niches)
        
        result = []
        for niche in niches:
            if len(result) >= n:
                break
            result.append(self.archive[niche])
        
        return result
    
    def save(self, path: Path) -> None:
        """Save archive to disk."""
        # Convert to JSON-serializable format
        serialized = {
            str(k): v for k, v in self.archive.items()
        }
        path.write_text(json.dumps(serialized))
    
    def load(self, path: Path) -> None:
        """Load archive from disk."""
        data = json.loads(path.read_text())
        self.archive = {eval(k): v for k, v in data.items()}
```

---

## 2.4 Integration with ML Loop

**File:** `backend/tools/ml_loop.py`

```python
def run_ml_loop(
    # ... existing params ...
    use_diversity: bool = True,
    archive_path: Path | None = None,
):
    # ... existing setup ...
    
    # Load or create diversity archive
    archive = DiversityArchive(NICHE_DIMENSIONS)
    if archive_path and archive_path.exists():
        archive.load(archive_path)
    
    for iteration in range(total_iterations):
        # Generate candidates
        run_result = run_generation(...)
        
        # Update archive with new verified builds
        for record in run_result["generation_records"]:
            if record.get("gate_pass"):
                archive.update(record)
        
        # Select diverse elites for next iteration
        if use_diversity:
            elites = select_diverse_elites(
                candidates,
                elite_count,
                archive=archive.archive,
            )
        else:
            elites = sorted(candidates, key=lambda c: c.score)[:elite_count]
        
        # ... rest of loop ...
    
    # Save archive
    if archive_path:
        archive.save(archive_path)
```

---

## 2.5 Implementation Tasks

| Task ID | Description | File | Effort |
|---------|------------|------|--------|
| DIV-01 | Define niche dimensions | diversity.py | 2 hr |
| DIV-02 | Implement assign_niche() | diversity.py | 2 hr |
| DIV-03 | Implement select_diverse_elites() | diversity.py | 4 hr |
| DIV-04 | Implement DiversityArchive class | archive.py | 4 hr |
| DIV-05 | Integrate with ml_loop.py | ml_loop.py | 2 hr |
| DIV-06 | Test: Verify diverse selection | - | 2 hr |
| DIV-07 | Test: Archive maintains best-per-niche | - | 2 hr |

---

# Part 3: Exploration Strategy

## 3.1 Problem Analysis

### Current: Pure Random
```python
# Random seed-based generation
candidate = generate_build(seed=random_seed)
```

This explores inefficiently - billions of possibilities, random sampling finds almost nothing.

### What We Need
- **Exploitation**: Use ML model to guide toward good builds
- **Exploration**: Don't just follow model - try new things
- **Novelty**: Reward builds that are different from seen ones

---

## 3.2 Solution: Multi-Strategy Exploration

### Strategy 1: Epsilon-Greedy

```python
def select_with_epsilon_greedy(
    candidates: list[Candidate],
    model: SurrogateModel,
    epsilon: float = 0.1,
) -> list[Candidate]:
    """Select candidates with ε-greedy strategy.
    
    Args:
        candidates: Available candidates
        model: Surrogate model for scoring
        epsilon: Probability of random selection (exploration)
        
    Returns:
        Selected candidates
    """
    n_random = max(1, int(len(candidates) * epsilon))
    n_exploit = len(candidates) - n_random
    
    # Random selection (exploration)
    random.shuffle(candidates)
    random_selection = candidates[:n_random]
    
    # Model-based selection (exploitation)
    scored = [(c, model.predict(c)) for c in candidates[n_random:]]
    exploit_selection = [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:n_exploit]]
    
    return random_selection + exploit_selection
```

### Strategy 2: Novelty Search

```python
def compute_novelty(
    build: dict[str, Any],
    population: list[dict[str, Any]],
    k: int = 5,
) -> float:
    """Compute novelty score based on distance to k-nearest neighbors.
    
    Novelty = average distance to k nearest neighbors in feature space
    """
    if not population:
        return 1.0  # Most novel if nothing to compare to
    
    # Feature vector
    features = build_to_features(build)
    pop_features = [build_to_features(p) for p in population]
    
    # Compute distances
    distances = [np.linalg.norm(features - pf) for pf in pop_features]
    distances.sort()
    
    # Average distance to k nearest
    k_distances = distances[:k]
    return sum(k_distances) / len(k_distances)

def build_to_features(build: dict[str, Any]) -> np.ndarray:
    """Convert build to feature vector for distance computation."""
    return np.array([
        build.get("full_dps", 0),
        build.get("max_hit", 0),
        build.get("life", 0),
        build.get("armour", 0),
        build.get("evasion", 0),
        # ... more features
    ])
```

### Strategy 3: Curiosity-Driven

```python
class CuriosityExploration:
    """Explore areas where model performs poorly."""
    
    def __init__(self, model: SurrogateModel):
        self.model = model
        self.error_history: list[dict[str, Any]] = []
    
    def update(self, build: dict[str, Any], actual: float, predicted: float):
        """Record prediction error for this build."""
        error = abs(actual - predicted)
        self.error_history.append({
            "build": build,
            "error": error,
            "features": build_to_features(build),
        })
    
    def get_target_regions(self, n: int = 3) -> list[dict]:
        """Get regions of feature space where model is weakest."""
        if not self.error_history:
            return []
        
        # Find builds with highest error
        self.error_history.sort(key=lambda x: x["error"], reverse=True)
        
        # Cluster high-error builds to find regions
        high_error = [e["features"] for e in self.error_history[:50]]
        
        if len(high_error) < n:
            return []
        
        # Simple clustering
        clusters = np.array_split(high_error, n)
        return [{
            "center": cluster.mean(axis=0),
            "size": len(cluster),
        } for cluster in clusters]
    
    def score_candidate(
        self, 
        candidate: Candidate,
        base_score: float,
    ) -> float:
        """Score candidate with curiosity bonus for unexplored regions."""
        features = build_to_features(candidate.to_dict())
        
        # How close to high-error regions?
        bonus = 0.0
        for region in self.get_target_regions():
            distance = np.linalg.norm(features - region["center"])
            bonus += 1.0 / (1.0 + distance)
        
        return base_score + bonus * 0.1  # 10% curiosity weight
```

### Strategy 4: Pareto Multi-Objective

```python
def select_pareto_frontier(
    candidates: list[Candidate],
    objectives: list[str] = ["full_dps", "max_hit", "life"],
) -> list[Candidate]:
    """Select Pareto-optimal candidates across multiple objectives."""
    
    def dominates(a: dict, b: dict) -> bool:
        """A dominates B if A is >= in all objectives and > in at least one."""
        better_in_any = False
        for obj in objectives:
            if a.get(obj, 0) < b.get(obj, 0):
                return False
            if a.get(obj, 0) > b.get(obj, 0):
                better_in_any = True
        return better_in_any
    
    pareto = []
    for candidate in candidates:
        c_dict = candidate.to_dict()
        dominated = False
        for existing in pareto:
            if dominates(existing, c_dict):
                dominated = True
                break
        if not dominated:
            # Remove any that this candidate dominates
            pareto = [p for p in pareto if not dominates(c_dict, p.to_dict())]
            pareto.append(candidate)
    
    return pareto
```

---

## 3.3 Unified Selection Algorithm

**File:** `backend/engine/generation/exploration.py`

```python
@dataclass
class ExplorationConfig:
    """Configuration for exploration strategy."""
    epsilon: float = 0.1          # Random exploration rate
    novelty_weight: float = 0.2    # Weight for novelty bonus
    curiosity_weight: float = 0.1 # Weight for curiosity bonus
    use_pareto: bool = True       # Use Pareto selection
    
    # Decay schedule
    epsilon_decay: float = 0.95   # Multiply epsilon each iteration
    min_epsilon: float = 0.01     # Minimum exploration rate

def select_candidates(
    candidates: list[Candidate],
    model: SurrogateModel | None,
    config: ExplorationConfig,
    archive: DiversityArchive | None = None,
    curiosity: CuriosityExploration | None = None,
    iteration: int = 0,
) -> list[Candidate]:
    """Unified candidate selection with exploration.
    
    Combines:
    - Epsilon-greedy random exploration
    - Model-based exploitation  
    - Novelty search for diversity
    - Curiosity-driven targeting of weak areas
    - Pareto frontier for multi-objective
    """
    
    # Apply epsilon decay
    epsilon = max(config.min_epsilon, config.epsilon * (config.epsilon_decay ** iteration))
    
    # Score all candidates
    scored = []
    for candidate in candidates:
        base_score = model.predict(candidate) if model else candidate.score
        
        # Add novelty bonus
        if config.novelty_weight > 0 and archive:
            novelty = compute_novelty(candidate.to_dict(), list(archive.archive.values()))
            base_score += novelty * config.novelty_weight
        
        # Add curiosity bonus
        if config.curiosity_weight > 0 and curiosity:
            base_score = curiosity.score_candidate(candidate, base_score)
        
        scored.append((candidate, base_score))
    
    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Epsilon-greedy selection
    n_random = max(1, int(len(scored) * epsilon))
    random.shuffle(scored)
    random_candidates = [c for c, _ in scored[:n_random]]
    exploit_candidates = [c for c, _ in scored[n_random:]]
    
    # Pareto filter if enabled
    if config.use_pareto:
        exploit_candidates = select_pareto_frontier(exploit_candidates)
    
    return random_candidates + exploit_candidates
```

---

## 3.4 Implementation Tasks

| Task ID | Description | File | Effort |
|---------|------------|------|--------|
| EXP-01 | Implement epsilon_greedy selection | exploration.py | 2 hr |
| EXP-02 | Implement novelty search | exploration.py | 4 hr |
| EXP-03 | Implement CuriosityExploration class | exploration.py | 4 hr |
| EXP-04 | Implement Pareto frontier selection | exploration.py | 2 hr |
| EXP-05 | Implement unified select_candidates() | exploration.py | 4 hr |
| EXP-06 | Add ExplorationConfig to ml_loop | ml_loop.py | 2 hr |
| EXP-07 | Test: Verify exploration behavior | - | 4 hr |
| EXP-08 | Tune: Find good epsilon/weights | - | 8 hr |

---

# Part 4: Integration & Testing

## 4.1 Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML LOOP                                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. Load archive (best-per-niche from previous runs)               │
│  2. For each iteration:                                             │
│     a. Generate candidates (random + model-based)                  │
│     b. Evaluate ALL in PoB (pass OR fail)                          │
│     c. Update diversity archive with passing builds                │
│     d. Build dataset: ALL builds (pass + fail)                     │
│     e. Train: classifier (pass/fail) + regressor (DPS)             │
│     f. Select diverse elites (MAP-Elites)                          │
│     g. Apply exploration strategy (epsilon/novelty/curiosity)      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     ML DATASET                                      │
├─────────────────────────────────────────────────────────────────────┤
│  • gate_pass: 1 or 0 (binary)                                      │
│  • gate_fail_reasons: list of strings                              │
│  • All metrics (DPS, defense, etc.)                               │
│  • Niche assignment (class, damage, defense, skill type)          │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2 Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| `pass_rate` | % of builds passing gates | Increase over time |
| `archive_size` | Total niches filled | Grow to fill ~50+ niches |
| `diversity_score` | Avg distance between archive builds | High = diverse |
| `exploration_rate` | % of candidates from random | Decrease over time |
| `novelty_score` | Avg novelty of new builds | Stable or increasing |
| `model_accuracy` | Classifier accuracy on pass/fail | Increase over time |

---

# Part 5: Implementation Order

## Phase 1: Failure Learning (Week 1)
```
Priority: CRITICAL - unblocks ML training

Day 1-2: FL-01, FL-02 (Don't delete failed builds)
Day 3:   FL-03 (Include gate_pass in dataset)
Day 4:   FL-04 (Train on failures)
Day 5:   FL-05, FL-06 (Testing)
```

## Phase 2: Diversity (Week 2)
```
Day 1-2: DIV-01, DIV-02 (Niche assignment)
Day 3:   DIV-03 (Diverse selection)
Day 4:   DIV-04 (DiversityArchive)
Day 5:   DIV-05 (Integrate with ml_loop)
Day 6-7: DIV-06, DIV-07 (Testing)
```

## Phase 3: Exploration (Week 3)
```
Day 1-2: EXP-01, EXP-02 (Epsilon + Novelty)
Day 3:   EXP-03 (Curiosity)
Day 4:   EXP-04 (Pareto)
Day 5:   EXP-05 (Unified selection)
Day 6:   EXP-06 (Integration)
Day 7-8: EXP-07, EXP-08 (Testing + Tuning)
```

## Phase 4: Full Integration (Week 4)
```
Day 1-2: Connect all components
Day 3-4: End-to-end test
Day 5:   Tune hyperparameters
Day 6-7: Documentation
```

---

# Appendix A: File Changes Summary

## Modified Files

| File | Changes |
|------|---------|
| `backend/engine/generation/runner.py` | Don't delete failed builds, keep artifacts |
| `backend/engine/surrogate/dataset.py` | Include gate_pass in rows |
| `backend/engine/surrogate/model.py` | Train classifier + regressor |
| `backend/tools/ml_loop.py` | Integrate diversity + exploration |

## New Files

| File | Purpose |
|------|---------|
| `backend/engine/generation/diversity.py` | Niche assignment, diverse selection |
| `backend/engine/archive/diversity_archive.py` | MAP-Elites archive |
| `backend/engine/generation/exploration.py` | All exploration strategies |

---

# Appendix B: Acceptance Criteria

## Phase 1: Failure Learning
- [ ] Failed builds persist in DB after evaluation
- [ ] `data/builds/` contains failed build artifacts
- [ ] Dataset includes gate_pass column
- [ ] Dataset includes gate_fail_reasons
- [ ] Train function accepts include_failures=True
- [ ] Classifier achieves >60% accuracy on pass/fail prediction

## Phase 2: Diversity
- [ ] Niche dimensions defined (class, damage, defense, skill)
- [ ] assign_niche() correctly categorizes builds
- [ ] select_diverse_elites() returns diverse set
- [ ] DiversityArchive maintains best-per-niche
- [ ] Archive contains >20 filled niches after 10 iterations

## Phase 3: Exploration
- [ ] Epsilon-greedy selection works
- [ ] Novelty search rewards different builds
- [ ] Curiosity targets weak model areas
- [ ] Pareto selection returns multi-objective optimal
- [ ] Exploration rate decreases over iterations

## Full Integration
- [ ] ML loop runs end-to-end
- [ ] Pass rate increases over iterations
- [ ] Archive maintains diverse builds
- [ ] Model accuracy improves over time
