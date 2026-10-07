from pathlib import Path
import pygame as pg

BG = (18, 18, 18)
INK = (239, 238, 233)
MUTED = (154, 154, 151)
RULE = (65, 65, 63)
SIZE = (1440, 960)


class Painter:
    def __init__(self, surface, scale=1):
        self.surface = surface
        self.scale = scale
        self.fonts = {}

    def point(self, point):
        return tuple(round(v*self.scale) for v in point)

    def rect(self, color, rect, width=0):
        r = pg.Rect(rect)
        left, top = self.point(r.topleft)
        right, bottom = self.point(r.bottomright)
        target = pg.Rect(left, top, right-left, bottom-top)
        pg.draw.rect(self.surface, color, target, max(1, round(width*self.scale)) if width else 0)

    def segment(self, color, start, end, width=1):
        pg.draw.line(self.surface, color, self.point(start), self.point(end), max(1, round(width*self.scale)))

    def circle(self, color, center, radius, width=0):
        pg.draw.circle(self.surface, color, self.point(center), max(1, round(radius*self.scale)),
                       max(1, round(width*self.scale)) if width else 0)

    def piece(self, piece, size, position):
        self.surface.blit(piece_image(piece, max(1, round(size*self.scale))), self.point(position))

    def text(self, text, x, y, size=18, color=INK, mono=False):
        key = size, mono
        if key not in self.fonts:
            self.fonts[key] = pg.font.Font(str(Path(__file__).parent / 'assets' / ('Mono.ttf' if mono else 'Sans.ttf')), max(1, round(size*self.scale)))
            if not mono:
                self.fonts[key].set_bold(True)
        self.surface.blit(self.fonts[key].render(str(text), True, color), self.point((x, y)))

    def line(self, x, y, w, color=RULE):
        self.segment(color, (x, y), (x+w, y))

    def label(self, text, x, y):
        self.text(text.upper(), x, y, 13, MUTED, True)


_pieces = {}


def piece_image(piece, size):
    """Original silhouettes on a 100-unit canvas, smoothly scaled for the board."""
    key = piece.symbol(), size
    if key in _pieces:
        return _pieces[key]
    s = pg.Surface((100, 100), pg.SRCALPHA)
    fill, edge = ((246, 245, 240), (28, 28, 28)) if piece.color else ((30, 30, 30), (229, 228, 222))
    def poly(points):
        pg.draw.polygon(s, fill, points)
        pg.draw.polygon(s, edge, points, 2)
    def circle(center, radius):
        pg.draw.circle(s, fill, center, radius)
        pg.draw.circle(s, edge, center, radius, 2)
    t = piece.piece_type
    if t == 1:
        poly([(37, 49), (63, 49), (59, 63), (69, 77), (31, 77), (41, 63)])
        circle((50, 35), 14)
    elif t == 2:
        poly([(28, 76), (34, 62), (54, 46), (46, 42), (30, 54), (21, 44), (36, 24), (43, 15), (48, 25), (61, 19), (73, 37), (73, 76)])
        pg.draw.circle(s, edge, (45, 33), 3)
    elif t == 3:
        poly([(28, 76), (42, 58), (39, 50), (29, 41), (35, 27), (50, 12), (65, 27), (71, 41), (61, 50), (58, 58), (72, 76)])
        pg.draw.line(s, edge, (54, 24), (43, 40), 3)
    elif t == 4:
        poly([(28, 76), (34, 43), (25, 38), (25, 19), (37, 19), (37, 29), (44, 29), (44, 19), (56, 19), (56, 29), (63, 29), (63, 19), (75, 19), (75, 38), (66, 43), (72, 76)])
    elif t == 5:
        poly([(30, 75), (36, 56), (24, 30), (40, 42), (50, 22), (60, 42), (76, 30), (64, 56), (70, 75)])
        for center in [(24, 27), (50, 20), (76, 27)]:
            circle(center, 5)
    else:
        poly([(31, 76), (39, 57), (30, 40), (33, 31), (67, 31), (70, 40), (61, 57), (69, 76)])
        poly([(46, 31), (46, 23), (39, 23), (39, 16), (46, 16), (46, 9), (54, 9), (54, 16), (61, 16), (61, 23), (54, 23), (54, 31)])
    poly([(29, 76), (71, 76), (77, 87), (23, 87)])
    result = pg.transform.smoothscale(s, (size, size))
    _pieces[key] = result
    return result
