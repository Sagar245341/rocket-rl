# Measured controller results

| Phase | Scenario | Controller | Episodes | Safe landings (95% CI) | Mean return +/- std | Successful-flight fuel |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | nominal | guidance | 100 | 100% (96.3%-100.0%) | 79.45 +/- 1.63 | 16.89 |
| 1 | nominal | ppo | 100 | 100% (96.3%-100.0%) | 83.24 +/- 1.62 | 15.21 |
| 1 | nominal | random | 100 | 2% (0.6%-7.0%) | -166.38 +/- 55.44 | 16.78 |
| 2 | nominal | guidance | 100 | 100% (96.3%-100.0%) | 83.42 +/- 3.12 | 17.32 |
| 2 | nominal | ppo | 100 | 100% (96.3%-100.0%) | 86.39 +/- 3.32 | 16.07 |
| 2 | nominal | random | 100 | 0% (0.0%-3.7%) | -174.40 +/- 40.07 | N/A |
| 3 | nominal | guidance | 100 | 100% (96.3%-100.0%) | 82.92 +/- 3.83 | 18.10 |
| 3 | nominal | ppo | 100 | 99% (94.6%-99.8%) | 82.56 +/- 19.78 | 17.37 |
| 3 | nominal | random | 100 | 0% (0.0%-3.7%) | -151.44 +/- 34.82 | N/A |
| 3 | wind_2x | guidance | 100 | 95% (88.8%-97.8%) | 70.90 +/- 43.93 | 18.76 |
| 3 | wind_2x | ppo | 100 | 91% (83.8%-95.2%) | 58.23 +/- 63.83 | 20.02 |
| 3 | wind_2x | random | 100 | 0% (0.0%-3.7%) | -151.46 +/- 27.46 | N/A |
| 3 | delay_2x | guidance | 100 | 100% (96.3%-100.0%) | 82.27 +/- 3.85 | 18.36 |
| 3 | delay_2x | ppo | 100 | 99% (94.6%-99.8%) | 82.47 +/- 19.80 | 17.41 |
| 3 | delay_2x | random | 100 | 0% (0.0%-3.7%) | -138.52 +/- 31.00 | N/A |
| 3 | noise_3x | guidance | 100 | 100% (96.3%-100.0%) | 83.73 +/- 3.78 | 17.77 |
| 3 | noise_3x | ppo | 100 | 99% (94.6%-99.8%) | 82.58 +/- 19.75 | 17.36 |
| 3 | noise_3x | random | 100 | 0% (0.0%-3.7%) | -151.44 +/- 34.82 | N/A |
| 3 | combined | guidance | 100 | 95% (88.8%-97.8%) | 71.18 +/- 43.94 | 18.64 |
| 3 | combined | ppo | 100 | 92% (85.0%-95.9%) | 60.56 +/- 60.01 | 20.02 |
| 3 | combined | random | 100 | 0% (0.0%-3.7%) | -141.00 +/- 25.84 | N/A |

## Training and checkpoint selection

Phase 1: behavior cloning + PPO; selected checkpoint at 50,000 PPO steps of 100,352 trained.
Phase 2: behavior cloning + PPO; selected checkpoint at 75,000 PPO steps of 100,352 trained.
Phase 3: behavior cloning + PPO; selected checkpoint at 100,000 PPO steps of 100,352 trained.

A checkpoint selected at step 0 is the imitation warm start before PPO updates; it must not be described as a policy improved by PPO.

Results use paired initial-condition seeds and identical reward weights across controllers. Benchmark seeds are separate from training and checkpoint-selection seeds. Each PPO phase uses one training seed; confidence intervals describe evaluated episodes for that fixed policy, not variability across retraining. Successful-flight fuel is conditional on landing and can involve different subsets of scenarios. Stress tests are distribution shifts, not aerospace safety certification. Zero observed failures does not establish a 100% population success rate.

Raw data: [episodes.csv](episodes.csv), [summary.csv](summary.csv), [manifest.json](manifest.json).

Open [dashboard.html](dashboard.html) locally for interactive filtering.
