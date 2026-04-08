# TMC and Breaking Point Investigation

## TMC = "Too Many Crossings"
- NOT a real knot type -- means topoly couldn't resolve (>15 crossings after reduction)
- 65/89 knotted proteins had TMC in their topology dict (alongside real types)
- 19/89 had ONLY TMC (no resolved knot type at all)
- For the paper: report 70/100 with resolved knot types, note 19 as "complex/unresolved"

## Corrected knot type distribution (excluding TMC):
3_1 (trefoil): 33, 4_1 (figure-eight): 10, 5_2: 10, 3_1#3_1 (composite): 9,
5_1: 7, 6_2: 3, 7_6: 3, 8_19: 3, 6_1: 2, plus 6 rare types

## Breaking Point (90% median)
- Distribution: 5 at 10%, 2 at 30%, 2 at 40%, 5 at 50%, 4 at 60%, 20 at 70%,
  35 at 80%, 43 at 85%, 77 at 90%, 57 at 95% (never broke)
- Median=90%, Mean=83.5%
- The 90% is inflated because our grid has a 5% gap (85->90), and 57 proteins
  never broke even at 90% so they're counted as 95%
- More honest: "mean breaking point of 84%, with most proteins breaking between 80-90%"
- Or report both: "median 90%, mean 84%"
