import json

from stateaware.cli import main


def test_compare_command_does_not_require_environment_file(tmp_path, monkeypatch, capsys) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "trajectories": [
                    {"case_id": "case-1", "condition": "direct", "last_turn_asr": 0},
                    {"case_id": "case-1", "condition": "ra_fixed", "last_turn_asr": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    def fail_if_settings_are_loaded(*args, **kwargs):
        raise AssertionError("compare should not load environment settings")

    monkeypatch.setattr("stateaware.cli.Settings.load", fail_if_settings_are_loaded)

    result = main(
        [
            "compare",
            "--summary",
            str(summary_path),
            "--baseline",
            "direct",
            "--candidate",
            "ra_fixed",
            "--metric",
            "last_turn_asr",
            "--bootstrap-samples",
            "100",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["paired_cases"] == 1
