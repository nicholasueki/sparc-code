"""Hardware-free tests for model backend resource lifecycles."""
import base64
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "node_c"))

from sparc_node_c import backends  # noqa: E402


IMAGE_B64 = base64.b64encode(b"jpeg bytes").decode()


def _backend(*, apply_chat_template=None, generate=None):
    backend = backends.MLXBackend.__new__(backends.MLXBackend)
    backend.model = object()
    backend.processor = object()
    backend.config = {}
    backend._apply_chat_template = apply_chat_template or (
        lambda processor, config, messages, num_images: "prompt"
    )
    backend._generate = generate or (
        lambda model, processor, prompt, **kwargs: SimpleNamespace(text="answer")
    )
    return backend


@pytest.fixture
def opened_temp_files(tmp_path, monkeypatch):
    real_named_temporary_file = backends.tempfile.NamedTemporaryFile
    opened = []

    def tracked_named_temporary_file(*args, **kwargs):
        temp_file = real_named_temporary_file(*args, dir=tmp_path, **kwargs)
        opened.append(temp_file)
        return temp_file

    monkeypatch.setattr(
        backends.tempfile, "NamedTemporaryFile", tracked_named_temporary_file
    )
    return opened


def _assert_cleaned(opened_temp_files):
    assert len(opened_temp_files) == 1
    temp_file = opened_temp_files[0]
    assert temp_file.closed
    assert not Path(temp_file.name).exists()


def test_mlx_frame_temp_file_is_closed_and_removed_after_success(
    opened_temp_files, monkeypatch
):
    real_unlink = backends.unlink

    def assert_closed_then_unlink(path):
        assert opened_temp_files[0].closed
        real_unlink(path)

    monkeypatch.setattr(backends, "unlink", assert_closed_then_unlink)
    backend = _backend()

    assert backend.generate("system", "user", IMAGE_B64) == "answer"

    _assert_cleaned(opened_temp_files)


def test_mlx_frame_temp_file_is_cleaned_after_decode_failure(opened_temp_files):
    backend = _backend()

    with pytest.raises(ValueError):
        backend.generate("system", "user", "invalid-base64")

    _assert_cleaned(opened_temp_files)


def test_mlx_frame_temp_file_is_cleaned_after_template_failure(opened_temp_files):
    def fail_template(*args, **kwargs):
        raise RuntimeError("template failed")

    backend = _backend(apply_chat_template=fail_template)

    with pytest.raises(RuntimeError, match="template failed"):
        backend.generate("system", "user", IMAGE_B64)

    _assert_cleaned(opened_temp_files)


def test_mlx_frame_temp_file_is_cleaned_after_generation_failure(opened_temp_files):
    def fail_generation(*args, **kwargs):
        raise RuntimeError("generation failed")

    backend = _backend(generate=fail_generation)

    with pytest.raises(RuntimeError, match="generation failed"):
        backend.generate("system", "user", IMAGE_B64)

    _assert_cleaned(opened_temp_files)


def test_mlx_frame_temp_file_is_cleaned_after_base_exception(opened_temp_files):
    class Cancelled(BaseException):
        pass

    def cancel_generation(*args, **kwargs):
        raise Cancelled()

    backend = _backend(generate=cancel_generation)

    with pytest.raises(Cancelled):
        backend.generate("system", "user", IMAGE_B64)

    _assert_cleaned(opened_temp_files)


def test_mlx_unlink_failure_is_logged_without_masking_success(
    opened_temp_files, monkeypatch, caplog
):
    def fail_unlink(path):
        raise OSError("unlink failed")

    monkeypatch.setattr(backends, "unlink", fail_unlink)
    backend = _backend()

    with caplog.at_level(logging.ERROR, logger="sparc.backend"):
        assert backend.generate("system", "user", IMAGE_B64) == "answer"

    temp_file = opened_temp_files[0]
    assert temp_file.closed
    assert Path(temp_file.name).exists()
    assert str(temp_file.name) in caplog.text
    assert "failed to remove MLX frame temporary file" in caplog.text


def test_mlx_unlink_failure_is_logged_without_masking_generation_failure(
    opened_temp_files, monkeypatch, caplog
):
    original_error = RuntimeError("generation failed")

    def fail_generation(*args, **kwargs):
        raise original_error

    def fail_unlink(path):
        raise OSError("unlink failed")

    monkeypatch.setattr(backends, "unlink", fail_unlink)
    backend = _backend(generate=fail_generation)

    with caplog.at_level(logging.ERROR, logger="sparc.backend"):
        with pytest.raises(RuntimeError) as raised:
            backend.generate("system", "user", IMAGE_B64)

    temp_file = opened_temp_files[0]
    assert raised.value is original_error
    assert temp_file.closed
    assert Path(temp_file.name).exists()
    assert str(temp_file.name) in caplog.text
    assert "failed to remove MLX frame temporary file" in caplog.text
