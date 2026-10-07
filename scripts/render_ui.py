"""Capture honest UI previews without opening a window or starting training."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from pathlib import Path
import time
import pygame as pg
from fly_chess.ui.app import Application


def main():
    pg.init()
    root = Path.cwd()
    output = root / 'logs' / 'ui-preview'
    output.mkdir(parents=True, exist_ok=True)
    app = Application(root)
    try:
        for screen in ('menu', 'train', 'settings'):
            app.screen = screen
            pg.image.save(app.draw(), output / f'{screen}.png')
        app.new_game()
        deadline = time.monotonic()+45
        while app.data is None and not app.error and time.monotonic() < deadline:
            app.tick()
            time.sleep(.02)
        if app.error or app.data is None:
            raise RuntimeError(app.error or 'Inspection timed out')
        for tab in ('Overview', 'Layers', 'Opponent'):
            app.tab = tab
            pg.image.save(app.draw(), output / f'play-{tab.lower()}.png')
        import chess
        app.play_move(chess.Move.from_uci('e2e4'))
        while not app.decisions and not app.error and time.monotonic() < deadline:
            app.tick()
            time.sleep(.02)
        if app.error or not app.decisions:
            raise RuntimeError(app.error or 'Search timed out')
        app.step_record(-1)
        app.tab = 'Search'
        pg.image.save(app.draw(), output / 'play-search.png')
        pg.image.save(app.draw(1200/1440), output / 'play-minimum.png')
        print(output)
    finally:
        app.close()
        pg.quit()


if __name__ == '__main__':
    main()
