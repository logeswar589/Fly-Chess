import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame as pg
import pytest
from pathlib import Path
from fly_chess.ui import theme

from fly_chess.ui.app import Application
from fly_chess.ui.board import square_at, square_rect
from fly_chess.ui.theme import Painter, SIZE


@pytest.mark.parametrize('scale', [1200/1440, 1, 1920/1440, 2])
def test_native_render_and_logical_hit_targets(tmp_path, scale):
    pg.init()
    app = Application(tmp_path)
    try:
        for screen in ('menu', 'play', 'train', 'settings'):
            app.screen = screen
            surface = app.draw(scale)
            assert surface.get_size() == tuple(round(v*scale) for v in SIZE)
            expected = pg.font.Font(str(Path(theme.__file__).parent/'assets/Sans.ttf'), round(24*scale))
            assert app.p.fonts[(24, False)].get_height() == expected.get_height()
            assert app.buttons[0][0] == pg.Rect(1054, 27, 94, 38)
        center = square_rect(12).center
        assert square_at(tuple(v/scale for v in (center[0]*scale, center[1]*scale))) == 12
        app.draw()
        assert app.canvas.get_size() == SIZE
    finally:
        app.close()
        pg.quit()


def test_scaled_primitives_land_at_native_coordinates():
    surface = pg.Surface((200, 200))
    painter = Painter(surface, 2)
    painter.rect((255, 255, 255), (10, 20, 15, 10))
    assert surface.get_at((20, 40))[:3] == (255, 255, 255)
    assert surface.get_at((19, 40))[:3] == (0, 0, 0)
