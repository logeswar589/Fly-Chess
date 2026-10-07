"""Train → play → close → load → resume → evaluate → play from source or wheel."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

# Spawned workers inherit this exact package precedence, including on Windows.
if '--package-root' in sys.argv:
    sys.path.insert(0, str(Path(sys.argv[sys.argv.index('--package-root')+1]).resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import chess
    import pygame as pg
    import fly_chess
    from fly_chess.config import Config
    from fly_chess.ui.app import Application
    root = args.output or Path.cwd()/'logs'/('release-'+uuid4().hex[:8])
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.package_root:
        assert Path(fly_chess.__file__).resolve().is_relative_to(args.package_root.resolve())
    config = Config(device='cpu', network_channels=8, residual_blocks=1, selfplay_workers=2,
        batch_size=4, replay_buffer_size=64, mcts_simulations=2, max_game_plies=8,
        games_per_generation=2, updates_per_generation=2, evaluation_pairs=2,
        evaluation_min_pairs=20, evaluation_simulations=2, evaluation_max_plies=8)
    pg.init()
    app = None
    frame_times = []
    def wait(predicate, timeout=90):
        deadline = time.monotonic()+timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise RuntimeError('Release workflow timed out')
            started = time.monotonic()
            app.tick()
            app.draw()
            app.handle_event(pg.event.Event(pg.KEYDOWN, key=pg.K_TAB, mod=0))
            frame_times.append(time.monotonic()-started)
            if app.error or app.training.error:
                raise RuntimeError(app.error or app.training.error)
            time.sleep(.01)
        app.tick()
    try:
        app = Application(root, config=config)
        app.reduced_motion = True
        for screen in ('menu', 'train', 'settings'):
            app.screen = screen
            pg.image.save(app.draw(), root/f'{screen}-empty.png')
        app.screen = 'train'
        app.training.interval = 1
        app.training.module = 'stem.0'
        app.training.start()
        wait(lambda: not app.training.worker.busy)
        first = app.training.state['progress']
        assert first['completed_generations'] == 1 and first['training_steps'] == 2
        assert first['evaluation']['status'] == 'baseline'
        app.weights = root/'models/fly_latest.pt'
        app.difficulty = 0
        app.new_game()
        app.board_click(chess.E2)
        app.board_click(chess.E4)
        wait(lambda: len(app.session.game.board.move_stack) == 2)
        first_model = app.match_model_id
        assert first_model.endswith('step-2')
        pg.image.save(app.draw(), root/'first-play.png')
        app.close()
        app = Application(root)
        app.reduced_motion = True
        app.screen = 'train'
        app.training.start(load_only=True)
        wait(lambda: not app.training.worker.busy)
        assert app.training.state['progress']['training_steps'] == 2
        app.training.interval = 1
        app.training.module = 'stem.0'
        app.training.start()
        wait(lambda: not app.training.worker.busy)
        final = app.training.state['progress']
        assert final['completed_generations'] == 2 and final['training_steps'] == 4
        assert final['games_played'] == 4
        assert final['evaluation']['status'] == 'complete' and not final['evaluation']['promoted']
        for pane in ('Learning', 'Stats', 'Config', 'Human', 'Watch'):
            app.training.pane = pane
            pg.image.save(app.draw(), root/f'{pane.lower()}.png')
            pg.image.save(app.draw(1200/1440), root/f'{pane.lower()}-small.png')
            pg.image.save(app.draw(1920/1440), root/f'{pane.lower()}-large.png')
        app.weights = root/'models/fly_latest.pt'
        app.set_side(chess.BLACK)
        app.difficulty = 0
        app.new_game()
        wait(lambda: app.session.human_turn)
        assert app.match_model_id.endswith('step-4')
        assert app.match_model_id != first_model
        for tab in ('Overview', 'Layers', 'Search', 'Opponent'):
            app.tab = tab
            pg.image.save(app.draw(), root/f'play-{tab.lower()}.png')
        app.error = 'Verification error state / missing checkpoint'
        pg.image.save(app.draw(), root/'error-state.png')
        app.error = ''
        report = {'package': fly_chess.__file__, 'workspace': str(root),
            'first_model': first_model, 'resumed_model': app.match_model_id, 'progress': final,
            'frame_count': len(frame_times), 'mean_frame_ms': sum(frame_times)/len(frame_times)*1000,
            'max_frame_ms': max(frame_times)*1000, 'status': 'passed'}
        app.close()
        app = None
        import multiprocessing
        assert not multiprocessing.active_children(), 'Orphan training/self-play child processes'
        report['orphan_children'] = 0
        (root/'verification.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if app is not None:
            app.close()
        pg.quit()


if __name__ == '__main__':
    main()
