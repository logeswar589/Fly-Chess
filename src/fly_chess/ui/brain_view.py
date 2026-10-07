"""Interactive fly-inspired 3D layout of measured activation samples."""
import math
import pygame as pg
from fly_chess.ui.theme import INK, MUTED, RULE


class BrainView:
    def __init__(self):
        self.zoom = 1.
        self.yaw = .15
        self.pitch = -.12
        self.rect = pg.Rect(744, 320, 640, 365)
        self.drag = None
        self.phase = 'after'
        self.synapses = True

    def reset(self):
        self.zoom, self.yaw, self.pitch = 1., .15, -.12
        self.drag = None

    def handle(self, event, position):
        if event.type == pg.MOUSEBUTTONUP:
            self.drag = None
        if position is None:
            return False
        if event.type == pg.MOUSEWHEEL and self.rect.collidepoint(position):
            self.zoom = max(.6, min(4., self.zoom * 1.15**event.y))
            return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(position):
            self.drag = position
            return True
        if event.type == pg.MOUSEMOTION and self.drag is not None:
            if not event.buttons[0]:
                self.drag = None
                return False
            self.yaw += (position[0]-self.drag[0])*.008
            self.pitch = max(-1.3, min(1.3, self.pitch+(position[1]-self.drag[1])*.008))
            self.drag = position
            return True
        return False

    @staticmethod
    def nodes(snapshot):
        rows = []
        groups = (snapshot or {}).get('brain', {})
        grouped = {name: [] for name in ('stem', 'trunk', 'policy', 'value')}
        for name, tensor in groups.items():
            group = name.split('.')[0].split(':')[0]
            if group in grouped:
                for index, value in zip(tensor['indices'], tensor['values']):
                    if math.isfinite(value):
                        grouped[group].append((name, index, value))
        # Stable positions identify samples across before/after captures. Anatomical
        # placement is illustrative; no biological connectivity is asserted.
        for group, samples in grouped.items():
            for i, (name, index, value) in enumerate(samples):
                side = -1 if i % 2 == 0 else 1
                n = i//2
                count = max(1, (len(samples)+1)//2)
                z = 1-2*(n+.5)/count
                angle = n*2.399963229728653
                radius = math.sqrt(max(0, 1-z*z))
                depth = .52 + .48*((n*37 % 101)/100)**.3333
                u, v, w = radius*math.cos(angle)*depth, radius*math.sin(angle)*depth, z*depth
                if group == 'stem':
                    xyz = (side*(1.48+.39*u), .06+.66*v, .42*w)
                elif group == 'trunk':
                    xyz = (side*(.58+.67*u), -.02+.9*v, .64*w)
                elif group == 'policy':
                    xyz = (side*(.57+.38*u), -.67+.28*v, -.1+.35*w)
                else:
                    xyz = (side*(.17+.21*u), .84+.34*v, .32*w)
                rows.append((xyz, name, index, value))
        return rows

    def draw(self, p, snapshot, mouse, *, y=320, height=365, reference=None):
        self.rect = pg.Rect(744, y, 640, height)
        nodes = self.nodes(snapshot)
        if not nodes:
            p.text('Waiting for measured brain activity.', 764, y+110, 23)
            p.text('Enable diagnostics and activations in Settings / Config.', 764, y+150, 15, MUTED)
            return
        scales = {}
        for _, name, _, value in nodes + self.nodes(reference):
            scales[name] = max(scales.get(name, 1e-8), abs(value))
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        projected = []
        for (x, yy, z), name, index, value in nodes:
            x, z = x*cy+z*sy, -x*sy+z*cy
            yy, z = yy*cp-z*sp, yy*sp+z*cp
            unit = min(154, height/2.7)*self.zoom * 4/(4+z)
            projected.append((z, (1064+x*unit, y+height*.47+yy*unit), name, index, value))
        old_clip = p.surface.get_clip()
        p.surface.set_clip(pg.Rect(p.point(self.rect.topleft), p.point(self.rect.size)))
        hovered = None
        try:
            positions = {(name, index): point for _, point, name, index, _ in projected}
            edges = (snapshot or {}).get('connections', []) if self.synapses else []
            limit = max((abs(e['contribution']) for e in edges), default=1) or 1
            for edge in edges:
                a = positions.get((edge['source'], edge['source_index']))
                b = positions.get((edge['target'], edge['target_index']))
                if a and b:
                    strength = abs(edge['contribution'])/limit
                    gray = round(35+190*strength)
                    p.segment((gray, gray, gray), a, b, 2 if strength > .6 else 1)
            for z, point, name, index, value in sorted(projected, reverse=True):
                strength = abs(value)/scales[name]
                gray = round(45+strength*195)
                radius = max(1, (1.1+strength*1.9)*min(self.zoom, 2))
                if strength > .65:
                    p.circle((35, 35, 34), point, radius+2)
                p.circle((gray, gray, gray), point, radius, 0 if value >= 0 else 1)
                if self.rect.collidepoint(mouse) and math.dist(point, mouse) < 7:
                    hovered = (name, index, value, point)
            if hovered:
                p.circle(INK, hovered[3], 7, 1)
        finally:
            p.surface.set_clip(old_clip)
        p.text('STEM / INPUT', 752, y+height-23, 11, MUTED, True)
        p.text('TRUNK / POLICY / VALUE', 1074, y+height-23, 11, MUTED, True)
        detail = f'{hovered[0]} [{hovered[1]}] = {hovered[2]:+.5g}' if hovered else f'{len(nodes)} measured samples / {len(scales)} layer outputs'
        p.text(detail[:77], 744, y+height+5, 13, INK, True)
        p.text('Nodes: |activation| per layer / traces: |input × weight| / hollow: negative', 744, y+height+29, 12, MUTED)
        p.text('Fly-inspired layout / sampled artificial connections / not biological anatomy', 744, y+height+50, 12, MUTED)
