"""Finite real GUI/backend verification; writes only to a unique smoke workspace."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from pathlib import Path
import json
import time
from uuid import uuid4


def main():
    import pygame as pg
    from fly_chess.config import Config
    from fly_chess.ui.app import Application
    pg.init()
    root = Path.cwd() / 'logs' / ('gui-training-'+uuid4().hex[:8])
    screenshots = root / 'screenshots'
    screenshots.mkdir(parents=True)
    config = Config(device='cpu', network_channels=8, residual_blocks=1, mcts_simulations=2,
        games_per_generation=2, max_game_plies=8, updates_per_generation=2,
        batch_size=4, replay_buffer_size=16, evaluation_enabled=False)
    app = Application(root, config=config)
    try:
        app.screen = 'train'
        app.training.interval = 1
        app.training.module = 'stem.0'
        app.training.start()
        deadline = time.monotonic()+60
        durations = []
        while app.training.worker.busy and time.monotonic() < deadline:
            started = time.monotonic()
            app.tick()
            app.draw()
            app.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, mod=0))
            durations.append(time.monotonic()-started)
            time.sleep(.016)
        app.tick()
        if app.training.error or not app.training.state.get('inspection'):
            raise RuntimeError(app.training.error or 'No sampled update')
        for pane in ('Brain', 'Learning', 'Stats', 'Config', 'Watch'):
            app.training.pane = pane
            if pane == 'Watch':
                app.training.watch_playing = False
                app.training.manual_step(3)
                end = time.monotonic()+20
                while app.training.watch_data is None and time.monotonic() < end:
                    app.tick()
                    time.sleep(.02)
            pg.image.save(app.draw(), screenshots/f'{pane.lower()}.png')
            pg.image.save(app.draw(1200/1440), screenshots/f'{pane.lower()}-minimum.png')
        report = {'workspace': str(root), 'progress': app.training.state['progress'],
                  'frames_during_training': len(durations), 'max_frame_ms': max(durations)*1000,
                  'mean_frame_ms': sum(durations)/len(durations)*1000,
                  'last_inspected_step': app.training.state['inspection']['step']}
        (root/'verification.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        app.close()
        pg.quit()


if __name__ == '__main__':
    main()
