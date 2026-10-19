# Measured controller results

| Phase | Scenario | Controller | Episodes | Safe landings (95% CI) | Mean return +/- std | Successful-flight fuel |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | nominal | guidance | 100 | 100% (96.3%-100.0%) | 79.45 +/- 1.63 | 16.89 |
| 1 | nominal | ppo | 100 | 0% (0.0%-3.7%) | -210.56 +/- 4.74 | N/A |
| 1 | nominal | random | 100 | 2% (0.6%-7.0%) | -166.38 +/- 55.44 | 16.78 |

## Training and checkpoint selection

Phase 1: PPO; selected checkpoint at 50,000 PPO steps of 301,056 trained.

A checkpoint selected at step 0 is the imitation warm start before PPO updates; it must not be described as a policy improved by PPO.

Results use paired initial-condition seeds and identical reward weights across controllers. Benchmark seeds are separate from training and checkpoint-selection seeds. Each PPO phase uses one training seed; confidence intervals describe evaluated episodes for that fixed policy, not variability across retraining. Successful-flight fuel is conditional on landing and can involve different subsets of scenarios. Stress tests are distribution shifts, not aerospace safety certification. Zero observed failures does not establish a 100% population success rate.

Raw data: [episodes.csv](episodes.csv), [summary.csv](summary.csv), [manifest.json](manifest.json).

Open [dashboard.html](dashboard.html) locally for interactive filtering.
