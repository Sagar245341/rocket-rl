# Model card: guided rocket controllers

## Intended use

Educational simulation, reproducible controller comparison, and portfolio demonstration. The models control the simplified `RocketEnv-v2` environment. They are not suitable for real vehicles or safety-critical decisions.

## Architecture and training

Each model is a Stable-Baselines3 PPO `MlpPolicy` with separate feed-forward actor/critic networks. Policies consume eleven normalized observations and output three continuous actions. Each phase was trained independently with seed 42 and one CPU thread.

The guided configuration uses 32 reference-controller demonstration flights (seeds 1000-1031) and 32,000 sampled states labeled by that same controller, followed by 100,352 actual PPO environment steps. Actor warm-start training runs for 100 epochs. The reference controller is not used during neural-policy inference. The exact hyperparameters and installed library versions are included in each model's JSON metadata.

Checkpoint selection uses ten validation seeds (5000-5009), maximizing safe-landing rate and then mean return. Checkpoints at 0 PPO steps represent imitation alone. The shipped selected checkpoints are after 50,000 PPO environment steps for phase 1, 75,000 for phase 2, and 100,000 for phase 3; final policies are separately retained. Selection is based on validation only, not benchmark results.

## Evaluation

See [the benchmark report](benchmark/REPORT.md) for 100 paired held-out seeds per controller and condition, confidence intervals, returns, fuel metrics, and stress tests. The manifest records the exact checkpoint hashes. Nominal learned-policy safe landings were 100/100 in phase 1, 100/100 in phase 2, and 99/100 in phase 3. These are outcomes of fixed trained policies, not guaranteed probabilities.

## Known limitations

- One training seed per phase does not establish training reliability or algorithm-level superiority.
- Reference guidance provides privileged supervision, so this is not pure trial-and-error PPO.
- Only configured simulator distributions are evaluated. Stress tests do not establish real-world safety.
- Noise affects the first six normalized observations; several dynamics parameters remain directly observable.
- No drag, atmospheric variation, orbital mechanics, structural damage model, or detailed contact dynamics are simulated.
- The original PPO-only phase-1 run failed all 100 evaluation flights. It remains available as a documented baseline rather than being omitted.

## Files and provenance

`models/rocket_phaseN.zip` and its `.json` sidecar are the selected policy and metadata. `_final` files preserve the final policy. Monitor CSVs and validation JSONs support the published training plots. The selected files are byte-for-byte copies of the completed independent guided experiments under `models/augmented/`; this duplicate working directory is ignored by Git. The benchmark manifest records their original local paths and hashes; the same hashes identify the shipped files in `models/`.

Models and original project code are distributed under the repository's MIT license. Third-party libraries retain their own licenses.
