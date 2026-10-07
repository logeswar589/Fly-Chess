"""Real opt-in PGN → supervised candidate → conservative evaluation GUI smoke."""
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
    from fly_chess.neural.network import PolicyValueNetwork, NetworkSpec
    from fly_chess.neural.weights import save_weights
    from fly_chess.ui.app import Application
    pg.init()
    root = Path.cwd()/'logs'/('human-ui-'+uuid4().hex[:8])
    root.mkdir(parents=True)
    base = root/'models/fixture.pt'
    save_weights(PolicyValueNetwork(NetworkSpec(8, 1)), base, model_id='untrained-smoke-parent')
    config = Config(device='cpu', batch_size=4, replay_buffer_size=8, evaluation_pairs=1,
        evaluation_min_pairs=20, evaluation_simulations=2, evaluation_max_plies=4)
    app = Application(root, weights=base, config=config)
    try:
        app.new_game()
        app.save_human = True
        app.match_model_id = 'untrained-smoke-parent'
        for move in ('f2f3', 'e7e5', 'g2g4', 'd8h4'):
            if app.session.human_turn:
                app.session.move_human(move)
            else:
                app.session.apply_ai(app.session.token, move)
        assert app.session.finished
        app.new_game()
        for future, _, _ in app.archives.pending:
            future.result(timeout=5)
        assert not app.archives.drain()
        app.screen = 'train'
        app.training.pane = 'Human'
        app.draw()
        point = (850, 730)
        app.handle_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=point), point)
        assert app.training.worker.busy
        deadline = time.monotonic()+60
        frames = 0
        while app.training.worker.busy and time.monotonic() < deadline:
            app.tick()
            app.draw()
            frames += 1
            time.sleep(.02)
        app.tick()
        if app.training.error or not app.training.human_report:
            raise RuntimeError(app.training.error or 'Missing human report')
        report = app.training.human_report
        assert report['status'] == 'complete' and not report['evaluation']['promoted']
        pg.image.save(app.draw(), root/'human-training.png')
        pg.image.save(app.draw(1200/1440), root/'human-training-minimum.png')
        output = {'workspace': str(root), 'frames_during_training': frames, 'positions': report['positions'],
            'updates': len(report['updates']), 'status': report['status'], 'promoted': report['evaluation']['promoted'],
            'reason': report['evaluation']['reason'], 'candidate': report['candidate']}
        (root/'verification.json').write_text(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2))
    finally:
        app.close()
        pg.quit()


if __name__ == '__main__':
    main()
