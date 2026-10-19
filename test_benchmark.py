"""Statistics and scenario-regression checks for reproducible benchmarks."""
import unittest
from dataclasses import asdict
import numpy as np
from benchmark import summarize
from rocket_env import PhysicsConfig, RocketEnv


class BenchmarkTests(unittest.TestCase):
    def row(self,outcome='LANDED',fuel=20,reward=80):
        return dict(phase=3,scenario='nominal',controller='ppo',outcome=outcome,
                    fuel_used=fuel,reward=reward,duration_seconds=20)

    def test_wilson_interval_does_not_claim_certainty(self):
        result = summarize([self.row() for _ in range(100)])[0]
        self.assertEqual(result['success_rate'],1)
        self.assertLess(result['success_ci_low'],1)
        self.assertAlmostEqual(result['success_ci_high'],1)
        result = summarize([self.row('CRASHED') for _ in range(100)])[0]
        self.assertEqual(result['success_rate'],0)
        self.assertGreater(result['success_ci_high'],0)

    def test_fuel_report_is_conditional_on_success(self):
        result = summarize([self.row(fuel=20),self.row('CRASHED',fuel=2,reward=-100)])[0]
        self.assertEqual(result['mean_fuel'],11)
        self.assertEqual(result['successful_mean_fuel'],20)
        self.assertEqual(result['successes'],1)
        self.assertEqual(result['mean_reward'],-10)
        result = summarize([self.row('CRASHED')])[0]
        self.assertIsNone(result['successful_mean_fuel'])

    def test_phase_one_varies_and_repeats_by_seed(self):
        env = RocketEnv(1)
        first,_ = env.reset(seed=100)
        second,_ = env.reset(seed=101)
        replay,_ = env.reset(seed=100)
        self.assertFalse(np.array_equal(first,second))
        np.testing.assert_array_equal(first,replay)
        self.assertEqual(env.x,0)

    def test_stress_configs_share_initial_conditions(self):
        nominal = RocketEnv(3)
        stressed = RocketEnv(3,physics=PhysicsConfig(wind_scale=2,delay_scale=2,noise_scale=3))
        nominal.reset(seed=100)
        stressed.reset(seed=100)
        for attribute in ('x','y','vx','vy','angle','gravity','max_thrust'):
            self.assertEqual(getattr(nominal,attribute),getattr(stressed,attribute))
        self.assertAlmostEqual(stressed.wind,2*nominal.wind)
        self.assertAlmostEqual(stressed.engine_delay,2*nominal.engine_delay)

    def test_invalid_physics_rejected(self):
        for value in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                PhysicsConfig(wind_scale=value)
        self.assertEqual(asdict(PhysicsConfig())['noise_scale'],1)


if __name__ == '__main__':
    unittest.main()
