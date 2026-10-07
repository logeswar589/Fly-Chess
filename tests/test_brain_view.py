import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import pygame as pg
import torch
from fly_chess.neural.inspection import ActivationCapture, InspectionRequest
from fly_chess.neural.network import PolicyValueNetwork, NetworkSpec
from fly_chess.ui.brain_view import BrainView
from fly_chess.ui.theme import Painter


def test_brain_samples_match_every_leaf_output_without_changing_forward():
    model = PolicyValueNetwork(NetworkSpec(8, 1)).eval()
    states = torch.zeros(1, 21, 8, 8)
    expected = {}
    handles = [module.register_forward_hook(lambda m, i, out, name=name:
        expected.__setitem__(name+':0', out.detach().clone()))
        for name, module in model.named_modules() if not list(module.children())]
    try:
        with torch.no_grad():
            baseline = model(states)
            with ActivationCapture(model, InspectionRequest('brain', selected_module='stem', brain=True), {}) as capture:
                result = model(states)
        assert all(torch.equal(a, b) for a, b in zip(baseline, result))
        assert {k for k in capture.snapshot['brain'] if '/input:' not in k} == set(expected)
        for name, sample in capture.snapshot['brain'].items():
            if '/input:' not in name:
                assert sample['values'] == expected[name].reshape(-1)[sample['indices']].tolist()
        assert sum(len(s['values']) for s in capture.snapshot['brain'].values()) <= 8192
    finally:
        for handle in handles:
            handle.remove()
    assert all(not module._forward_hooks for module in model.modules())


def test_weighted_connections_are_actual_linear_and_convolution_terms():
    for layer, states in ((torch.nn.Linear(4, 3, bias=False), torch.tensor([[1., 2., 3., 4.]])),
                          (torch.nn.Conv2d(1, 1, 3, padding=1, bias=False), torch.arange(16.).reshape(1, 1, 4, 4))):
        model = torch.nn.Sequential(layer)
        with torch.no_grad():
            layer.weight.fill_(.1)
            if states.ndim == 4:
                layer.weight[0, 0, 1, 1] = -2
            else:
                layer.weight[:, 2] = -2
            with ActivationCapture(model, InspectionRequest('edges', brain=True), {}) as capture:
                model(states)
        edges = capture.snapshot['connections']
        assert edges
        for edge in edges:
            assert edge['input'] == states.reshape(-1)[edge['source_index']].item()
            assert edge['weight'] == -2
            assert edge['contribution'] == edge['input']*-2
            if states.ndim == 4:
                assert edge['source_index'] == edge['target_index']
            else:
                assert edge['source_index'] == 2


def test_brain_controls_and_empty_render_preserve_clip():
    pg.init()
    try:
        view = BrainView()
        point = view.rect.center
        for _ in range(50):
            view.handle(pg.event.Event(pg.MOUSEWHEEL, y=1), point)
        assert view.zoom == 4
        view.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1), point)
        view.handle(pg.event.Event(pg.MOUSEMOTION, buttons=(1, 0, 0)), (point[0]+20, point[1]+10))
        assert view.yaw > .15
        view.handle(pg.event.Event(pg.MOUSEBUTTONUP, button=1), None)
        assert view.drag is None
        surface = pg.Surface((1440, 960))
        clip = surface.get_clip()
        snap = {'brain': {'stem.0:0': {'indices': [0, 1, 2], 'values': [0., -2., 1.]}}}
        view.draw(Painter(surface), snap, point)
        assert surface.get_clip() == clip
        view.draw(Painter(surface), None, point)
        view.reset()
        assert view.zoom == 1
    finally:
        pg.quit()
