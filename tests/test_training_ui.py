import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pygame as pg
import pytest
import torch

from fly_chess.config import Config
from fly_chess.ui.training_worker import TrainingWorker
from fly_chess.ui.training_view import TrainingView, series_for
from fly_chess.ui.app import Application


def tiny_config():
    return Config(device='cpu', network_channels=8, residual_blocks=1, mcts_simulations=2,
        games_per_generation=2, max_game_plies=8, selfplay_workers=1, updates_per_generation=2,
        batch_size=4, replay_buffer_size=16, evaluation_enabled=False)


def wait_for(worker, predicate=lambda m: m.get('final'), timeout=45):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        for message in worker.drain():
            if message['kind'] == 'error':
                raise AssertionError(message['error'])
            if predicate(message):
                return message
        time.sleep(.02)
    raise AssertionError('Training worker timed out')


def close_worker(worker):
    worker.request('stop')
    deadline = time.monotonic()+45
    while worker.busy and time.monotonic() < deadline:
        worker.drain()
        time.sleep(.02)
    assert not worker.busy
    worker.cleanup()


def test_gui_bridge_pause_save_exit_reload_resume_with_real_captures(tmp_path):
    worker = TrainingWorker()
    config = tiny_config()
    try:
        assert worker.start(tmp_path, config, generations=10, interval=1, module='stem.0')
        initial = wait_for(worker, lambda m: m['progress']['status'] in ('ready', 'running'))
        worker.request('save')
        worker.request('pause')
        paused = wait_for(worker)
        assert paused['progress']['status'] == 'paused'
        assert Path(paused['latest']).exists()
        steps = paused['progress']['training_steps']
    finally:
        close_worker(worker)
    reopened = TrainingWorker()
    try:
        latest = tmp_path/'models/fly_latest.pt'
        reopened.start(tmp_path, config, resume=latest, load_only=True)
        loaded = wait_for(reopened, lambda m: m['progress']['status'] == 'loaded')
        assert loaded['progress']['training_steps'] == steps
        close_worker(reopened)
        reopened.start(tmp_path, config, resume=latest, interval=1, module='stem.0')
        final = wait_for(reopened)
        assert final['progress']['status'] == 'ready'
        assert final['progress']['training_steps'] > steps
        sample = final['inspection']
        assert sample['step'] == final['progress']['training_steps']
        assert sample['before']['tensors']['stem.0:0']['shape'] == [1, 8, 8, 8]
        assert sample['after']['tensors']['stem.0:0']['sample_values'] != sample['before']['tensors']['stem.0:0']['sample_values']
        assert sample['after']['brain']['stem.0:0']['values'] != sample['before']['brain']['stem.0:0']['values']
        assert 'policy.projection:0' in sample['after']['brain']
        assert 'value.6:0' in sample['after']['brain']
        assert sample['learning']['layers']['stem.0']['update_norm'] > 0
        assert final['watch']['status'] == 'truncated'
        assert final['metrics'][-1]['sample_keys'][0] == sample['sample_key']
        assert final['progress']['estimated_elo'] is None
    finally:
        close_worker(reopened)


def test_training_telemetry_does_not_change_weights_or_rng(tmp_path):
    from fly_chess.training.controller import TrainingController
    config = tiny_config()
    states = []
    for enabled in (False, True):
        with TrainingController(tmp_path/str(enabled), config) as controller:
            controller.inspection_interval_override = int(enabled)
            controller.activation_module = 'stem' if enabled else None
            controller.run()
            states.append({k: v.detach().clone() for k, v in controller.model.state_dict().items()})
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


def test_charts_keep_missing_evaluations_and_absolute_elo_missing():
    state = {'metrics': [{'step': 1, 'loss': 3., 'policy_loss': 2., 'value_loss': 1.,
                         'games_played': 2, 'average_game_length': 6.}],
             'evaluations': [{'generation': 1, 'games_played': 2, 'statistics': None},
                             {'generation': 2, 'games_played': 4, 'statistics': {'candidate_win_rate': .6}}]}
    series = series_for(state)
    assert series['win_rate'] == [[(1, None), (2, .6)]]
    assert series['rating'] == [[(2, None), (4, None)]]
    assert series['loss'][0] == [(1, 3.)]
    assert series['length'][0] == [(2, 6.)]


def test_watch_speed_steps_are_reversible_and_do_not_change_archive(tmp_path):
    view = TrainingView(tmp_path)
    record = {'root_fen': __import__('chess').STARTING_FEN, 'prefix_moves': [],
              'moves': ['e2e4', 'e7e5', 'g1f3'], 'game_id': 'test', 'model_id': 'test:step-0'}
    view.set_watch(record)
    initial = view.watch_game.board.fen()
    try:
        view.manual_step(2)
        assert view.watch_ply == 2 and not view.watch_playing
        view.manual_step(-2)
        assert view.watch_game.board.fen() == initial
        view.pane = 'Watch'
        view.speed_index = 4
        view.watch_playing = True
        view.tick()
        assert view.watch_ply == 3 and not view.brain.busy
        assert record['moves'] == ['e2e4', 'e7e5', 'g1f3']
    finally:
        view.close()


@pytest.mark.parametrize('speed_index', [0, 1, 2, 3])
def test_watch_timed_speeds_step_only_when_due(tmp_path, speed_index):
    from fly_chess.ui.training_view import SPEEDS
    view = TrainingView(tmp_path)
    view.set_watch({'root_fen': __import__('chess').STARTING_FEN, 'prefix_moves': [],
        'moves': ['e2e4', 'e7e5', 'g1f3'], 'game_id': 'test', 'model_id': 'test:step-0'})
    try:
        view.pane = 'Watch'
        view.speed_index = speed_index
        view.telemetry = False
        view.tick()
        assert view.watch_ply == 0
        view.last_step = time.monotonic()-1/SPEEDS[speed_index]-.01
        view.tick()
        assert view.watch_ply == 1
        view.tick()
        assert view.watch_ply == 1
    finally:
        view.close()


def test_gradient_inspection_can_be_disabled_independently(tmp_path):
    from fly_chess.training.controller import TrainingController
    with TrainingController(tmp_path, tiny_config()) as controller:
        controller.inspection_interval_override = 1
        controller.activation_module = 'stem'
        controller.inspect_gradients = False
        controller.run()
        inspection = controller.last_inspection
        assert inspection['after']['tensors']['stem:0']['sample_values']
        assert inspection['learning']['layers'] == {}
