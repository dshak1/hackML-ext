from pathlib import Path

from scripts.generate_demo_data import _assign_urgency, _make_frame
from scripts.train_model import _assert_input_files


def test_make_frame_has_required_columns() -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    df = _make_frame(50, start_id=1, rng=rng)
    expected = {
        "id",
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    }
    assert expected.issubset(set(df.columns))


def test_assign_urgency_range() -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    df = _make_frame(200, start_id=1, rng=rng)
    urgency = _assign_urgency(df, rng)
    assert urgency.between(0, 3).all()


def test_assert_input_files_message(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    train.write_text("id,urgency_level\n1,0\n", encoding="utf-8")

    missing_test = tmp_path / "test.csv"

    try:
        _assert_input_files(str(train), str(missing_test))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "generate_demo_data.py" in str(exc)
