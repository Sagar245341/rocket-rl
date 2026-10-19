"""Regression tests for CLI exports, checkpoint errors, and resized UI controls."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame
import app
import test as evaluation
from checkpoints import load_checkpoint
from rocket_env import ENV_VERSION, RocketEnv


class RuntimeTests(unittest.TestCase):
    def test_csv_with_nested_directory_and_spaces(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'flight results' / 'scores.csv'
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(['--headless','--episodes','2','--csv',str(target)])
            with target.open(newline='',encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r['seed'] for r in rows],['100','101'])
            self.assertTrue(all(r['outcome']=='LANDED' for r in rows))

    def test_invalid_csv_destination_has_readable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as error:
                evaluation.main(['--headless','--csv',folder])
            self.assertEqual(error.exception.code,2)
            self.assertIn('Cannot write CSV',output.getvalue())
            self.assertNotIn('Traceback',output.getvalue())

    def test_missing_model_does_not_overwrite_existing_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'scores.csv'
            target.write_text('preserve me',encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main(['--headless','--model',str(Path(folder)/'missing.zip'),'--csv',str(target)])
            self.assertEqual(target.read_text(encoding='utf-8'),'preserve me')

    def test_bad_checkpoint_metadata_and_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.zip'
            path.write_bytes(b'not a zip')
            with self.assertRaisesRegex(ValueError,'Missing metadata'):
                load_checkpoint(path,1)
            for content in ('[]','{broken',json.dumps({'environment':'LunarLander-v3'})):
                path.with_suffix('.json').write_text(content,encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_checkpoint(path,1)
            config = dict(environment=ENV_VERSION,phase=1,rewards=dict(landing=100,crash=-100,fuel=-2))
            path.with_suffix('.json').write_text(json.dumps(config),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Checkpoint phase'):
                load_checkpoint(path,2)
            with self.assertRaisesRegex(ValueError,'Cannot load PPO checkpoint'):
                load_checkpoint(path,1)
            config['rewards']['fuel'] = float('nan')
            path.with_suffix('.json').write_text(json.dumps(config),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'finite numbers'):
                load_checkpoint(path,1)

    def test_closing_visual_evaluation_stops_cleanly(self):
        original_render = RocketEnv.render
        def render_and_close(env):
            result = original_render(env)
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return result
        output = io.StringIO()
        with patch.object(RocketEnv,'render',render_and_close), contextlib.redirect_stdout(output):
            evaluation.main(['--episodes','1'])
        self.assertIn('Evaluation stopped',output.getvalue())
        self.assertIn('No completed flights',output.getvalue())
        pygame.quit()

    def test_resized_dashboard_mouse_controls(self):
        viewport = app.viewport_for((1024,600))
        # Click the Control button after resizing; it should restart in manual mode.
        point = (viewport.x+round(538*viewport.w/1400),viewport.y+round(738*viewport.h/970))
        batches = iter([
            [pygame.event.Event(pygame.VIDEORESIZE,w=1024,h=600,size=(1024,600))],
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=point)],
            [pygame.event.Event(pygame.KEYDOWN,key=pygame.K_p)],
            [pygame.event.Event(pygame.QUIT)],
        ])
        env = RocketEnv()
        with tempfile.TemporaryDirectory() as folder, patch.object(app,'ROOT',Path(folder)), \
             patch.object(app,'RocketEnv',return_value=env), \
             patch.object(env,'reset',wraps=env.reset) as reset, \
             patch.object(pygame.event,'get',side_effect=lambda: next(batches)):
            app.main()
        self.assertEqual(reset.call_count,2)
        self.assertEqual(env.steps,1)


if __name__ == '__main__':
    unittest.main()
