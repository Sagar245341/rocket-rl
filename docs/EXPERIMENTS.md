# Experiment protocol and limitations

## Reproduce the portfolio experiment

```powershell
.\venv\Scripts\python.exe train.py --config configs/guided_training.json --phase 1
.\venv\Scripts\python.exe train.py --config configs/guided_training.json --phase 2
.\venv\Scripts\python.exe train.py --config configs/guided_training.json --phase 3
.\venv\Scripts\python.exe benchmark.py
.\venv\Scripts\python.exe record_demo.py
```

Training commands above train each phase independently, with one CPU thread per process. Optional `--resume` supports curriculum experiments but is not used for the published guided runs. Outputs depend on the documented software versions and may vary slightly across platforms.

## Training and model selection

- Environment: `RocketEnv-v2`. All phases randomize initial altitude and vertical speed. Phase 1 constrains horizontal motion; phase 2 adds randomized horizontal position; phase 3 additionally randomizes other initial states and dynamics.
- Training seed: 42. Each benchmark manifest embeds exact training configuration, versions, checkpoint hash, and relevant source hashes.
- Pure PPO baseline: `configs/training.json`, 300,000 requested steps, no imitation. Actual steps round up to a complete 2,048-step rollout.
- Guided model: `configs/guided_training.json`, 100,000 requested PPO steps after behavior cloning. Each phase gets 32 reference flights (seeds 1000-1031), plus 32,000 expert-labeled sampled states. Augmentation decorrelates previous throttle from target throttle and exposes the network to states outside reference trajectories. Half the augmented states emphasize low altitude. These are supervised labels from the reference controller, not ground-truth optimal actions.
- Behavior cloning: 100 epochs of mean-squared action error, batch size 512, Adam learning rate 0.001, initial policy log standard deviation -2.0. PPO then uses learning rate 0.00003, gamma 0.9995, 2,048-step rollouts, and batches of 64. The reference controller is not called during learned-policy evaluation.
- Validation: fixed seeds 5000-5009, every 25,000 steps for guided runs and 50,000 for the raw baseline. Checkpoints are selected by safe-landing rate, with mean return as the tie-breaker. The initial imitation policy at step 0 is also eligible. Validation is used for selection, not for final reported performance.
- Selected and final checkpoints are both retained. A selected checkpoint at step 0 means PPO did not improve on the imitation warm start under the selection rule. Report it as an imitation-trained neural policy, not an RL improvement.

## Final evaluation

- 100 held-out initial-condition seeds, 10000-10099, for each controller, phase, and applicable condition.
- Controllers: learned checkpoint (`ppo` in artifact schemas), deterministic PD reference guidance, and independent uniform random actions in [-1,1]^3.
- Paired initial conditions: each controller sees the same scenario seed. Physics scales and reward weights are identical within each comparison. No test seed is used for imitation labels or checkpoint selection.
- Random actions use a separate generator seeded from action seed 98765, phase, and scenario seed. The action sequence is reused across physics interventions; it does not consume environment/sensor RNG.
- Outcomes use the actual touchdown thresholds in `rocket_env.py`; an arbitrary reward threshold is not treated as a safe landing.
- Safe-landing uncertainty: 95% Wilson binomial intervals. Return error bars show episode standard deviation, not confidence intervals. These statistics condition on one fixed trained policy and are not uncertainty across retraining seeds.
- Fuel: all-flight mean and successful-flight mean are both exported. Plots use successful-flight fuel; failed flights can terminate early and appear misleadingly fuel-efficient. Successful subsets can differ across controllers.

## Robustness interventions

Only phase 3 uses the advanced physics. The benchmark evaluates nominal conditions, twice nominal wind acceleration, twice engine delay, three times sensor noise, and their combination. Nominal ranges are wind acceleration -1.5 to +1.5 m/s2, engine lag time constant 0.15-0.4 seconds, and Gaussian standard deviation 0.002 in the first six normalized observations. Stress tests retain paired initial states and base random draws.

These are separate interventions, not a monotonic severity scale. Wind is fixed within an episode. Sensor draws share the same RNG seed, but different controller trajectories and durations can end at different times. Each flight is capped at 2,500 steps (50 simulated seconds).

## Limits of the evidence

This is a simplified 2D teaching environment, not an aerospace safety assessment. One training seed is insufficient to claim algorithm-level superiority. Reference-generated labels give guided policies privileged training knowledge and prevent a fair claim of learning entirely from trial and error. Benchmark seeds are fixed and visible, so future experiments should reserve a fresh final test set after further tuning. Do not claim generalization beyond the tested distributions or hide failed conditions.

The pure PPO baseline, failed preliminary guided runs, and final guided runs are retained locally under `models/baselines/`, `models/preliminary/`, and `models/augmented/` where applicable. Published per-episode data and manifests live under `docs/benchmark/`; the separate raw PPO comparison is under `docs/baselines/`.
