# Benchmarks (Modal A10G GPU, ESM3-SM 1.4B, fp16)

## Per-operation timings

| Operation | ~150 aa | ~250 aa | ~450 aa | Notes |
|-----------|---------|---------|---------|-------|
| Model load | 12s | - | - | One-time, cached on volume |
| Dataset load (HF) | 25s | - | - | 15k proteins, one-time |
| Structure gen (8 steps) | 0.6s* | 0.6s | 0.7s | *First call 9s (warmup) |
| Topoly (tries=30) | 2.2s | 2.1s | 2.6s | Roughly constant |
| Sequence gen (8 steps) | 0.6s | 0.5s | 0.7s | |
| Embedding extraction | 0.07s | 0.07s | 0.07s | Very fast |

## Full single trial (mask 10% -> seqgen -> structgen -> topoly)
**~3.3s per trial** for a ~250 aa protein

## Topoly validation on real knotted proteins
All 4 test proteins correctly detected as knotted (p >= 0.93):
- R_A0A7C7V383 (145 aa): 3_1 knot, knotted_p=0.93
- R_A0A1I0HZZ7 (244 aa): 3_1 knot, knotted_p=0.93
- R_A0A3A9GYT4 (258 aa): 3_1 knot, knotted_p=0.93
- R_A0A182LX58 (451 aa): 3_1 knot, knotted_p=1.00

## Parallel smoke test (5 proteins, 3 levels, 4 trials = 60 trials)
- Wall time: 66s (5 parallel containers)
- Total GPU time: 195s
- Per trial: 3.24s (confirms single-protein benchmark)
- Parallelism factor: 2.9x (limited by container startup overhead)

### Results confirm paper's claims directionally:
- At 25% masking: 4/5 proteins stay knotted (avg p > 0.7)
- At 50% masking: 3/5 proteins stay knotted
- At 75% masking: only 1/5 protein stays robustly knotted
- Pattern: knots ARE robust but break somewhere between 50-85%

## Revised cost estimates (A10G @ $1.10/hr)

### Masking stability sweep
Per protein: 12 trials at ~3.3s each per level
- 100 proteins x 10 levels x 8 trials = 8,000 trials
  ~26,400s GPU = ~7.3 GPU-hours = **~$8, ~45 min wall** (with 10 containers)
- 1000 proteins x 19 levels x 16 trials = 304,000 trials
  ~985,000s GPU = ~274 GPU-hours = **~$301, ~7.5 hrs wall** (with 10 containers)

### Embedding extraction
5000 proteins x 0.07s = ~6 min (trivial)
With masked variants (~400k): 400k x 0.07s = ~8 GPU-hours = ~$9

### Guided generation (de novo knotted protein)
Benchmarked: 3 attempts, length=256, 8 decoding steps, 10 samples/step
- All 3/3 produced knotted proteins (scores: 0.90, 1.00, 1.00)
- **~104s per attempt** (~1.7 min)
- ~12.7s per decoding step (10 samples scored with topoly each)
- For 1000 attempts: ~104,000s GPU = ~29 GPU-hours = **~$32**
- Wall time with 20 containers: ~1.5 hrs

### Unknotted-to-knotted
Similar to guided gen but iterative (up to 10 iterations per protein).
Rough estimate: ~5 iterations avg x ~104s = ~520s per protein
4000 proteins x 520s = ~578 GPU-hours = **~$636** (very expensive!)
Could reduce to 100 proteins pilot: ~14 GPU-hours = ~$16
