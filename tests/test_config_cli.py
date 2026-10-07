import json
import pytest

from fly_chess.cli import main
from fly_chess.config import Config, ConfigurationError, load_config


@pytest.mark.parametrize("values", [
    {"device": "magic"}, {"batch_size": 0}, {"mcts_simulations": True},
    {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
    {"learning_rate": -1}, {"selfplay_workers": 1.5}, {"seed": -1},
    {"claim_draws": "true"}, {"batch_size": 50, "replay_buffer_size": 10},
    {"network_channels": 0}, {"residual_blocks": 17}, {"cpu_threads": False},
    {"deterministic": "yes"},
])
def test_invalid_config(values):
    with pytest.raises(ConfigurationError):
        Config(**values)


def test_config_file_errors(tmp_path):
    path = tmp_path / "settings.toml"
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_config(path)
    path.write_text("batch_size = [", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_config(path)
    path.write_text('batch_siz = 5', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unknown settings: batch_siz"):
        load_config(path)
    path.write_text('device = "cpu"\nbatch_size = 16', encoding="utf-8")
    assert load_config(path).batch_size == 16


def test_diagnostic_creates_workspace_and_reports_rules(tmp_path, capsys):
    assert main(["--workspace", str(tmp_path), "diagnose"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["legal_action_count"] == report["mask_count"] == 20
    assert report["input_shape"] == [21, 8, 8]
    assert report["result"] is None
    for relative in ("data/selfplay", "data/human_games", "data/replay_buffer", "models", "logs"):
        assert (tmp_path / relative).is_dir()
    assert "Diagnostic passed" in (tmp_path / "logs/fly_chess.log").read_text()


def test_cli_bad_config_has_actionable_error(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text("selfplay_workers = 0", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(config), "diagnose"])
    assert exc.value.code == 2
    assert "selfplay_workers must be a positive integer" in capsys.readouterr().err
