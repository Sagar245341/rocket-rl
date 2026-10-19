"""Physics, reward, and reproducibility checks for RocketEnv."""
import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from rocket_env import RocketEnv, RewardConfig, guidance


class RocketTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        for phase in (1,2,3):
            check_env(RocketEnv(phase=phase), skip_render_check=True)

    def test_seeded_advanced_dynamics(self):
        a, b = RocketEnv(3), RocketEnv(3)
        np.testing.assert_array_equal(a.reset(seed=7)[0],b.reset(seed=7)[0])
        for _ in range(40):
            action = np.array([0.1,0.2,-0.3],dtype=np.float32)
            aa, bb = a.step(action), b.step(action)
            np.testing.assert_array_equal(aa[0],bb[0])
            self.assertEqual(aa[1:],bb[1:])

    def test_freefall_and_empty_tank(self):
        env = RocketEnv()
        env.fuel = 0
        before = env.vy
        env.step([1,0,0])
        self.assertAlmostEqual(env.vy,before-env.gravity*env.dt)
        self.assertEqual(env.throttle,0)
        self.assertEqual(env.fuel,0)
        self.assertEqual(env.x,0)

    def test_touchdown_thresholds(self):
        for velocity,offset,tilt,expected in [(-1,0,0,'LANDED'),(-8,0,0,'CRASHED'),
                                            (-1,20,0,'CRASHED'),(-1,0,.4,'CRASHED')]:
            env = RocketEnv(2)
            env.reset(seed=1)
            env.y,env.vy,env.x,env.angle = .001,velocity,offset,tilt
            _,_,terminated,_,info = env.step([-1,0,0])
            self.assertTrue(terminated)
            self.assertEqual(info['outcome'],expected)
            with self.assertRaises(RuntimeError):
                env.step([-1,0,0])

    def test_reward_weights_affect_only_scoring(self):
        a = RocketEnv(rewards=RewardConfig(fuel=-2))
        b = RocketEnv(rewards=RewardConfig(fuel=-5))
        a.reset(seed=1)
        b.reset(seed=1)
        aa, bb = a.step([1,0,0]), b.step([1,0,0])
        np.testing.assert_array_equal(aa[0],bb[0])
        self.assertAlmostEqual(bb[1]-aa[1],-3*aa[4]['fuel_used'])

    def test_guidance_lands_all_phases(self):
        for phase in (1,2,3):
            env = RocketEnv(phase)
            obs,_ = env.reset(seed=100)
            for _ in range(env.max_steps):
                obs,_,done,truncated,info = env.step(guidance(obs,phase))
                if done or truncated:
                    break
            self.assertTrue(info['is_success'],f'phase {phase}: {info}')

    def test_engine_lag_and_episode_limit(self):
        env = RocketEnv(3)
        env.reset(seed=7)
        env.step([1,0,0])
        self.assertGreater(env.throttle,0)
        self.assertLess(env.throttle,1)
        env.max_steps = env.steps + 1
        _,_,terminated,truncated,info = env.step([-1,0,0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['outcome'],'TIME LIMIT')


if __name__ == '__main__':
    unittest.main()
