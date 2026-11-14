# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Training must close its W&B run explicitly instead of leaving it to ``Tracking.__del__``.

Older runs relied on ``Tracking.__del__`` to call ``wandb.finish()``. Inside the Ray TaskRunner that only ran during
process teardown, after the wandb backend process had already exited. The finish call failed, and each offline run
lost the last unflushed 32 KiB block of its transaction log, which held the final training step(s) and the exit
record. Synced to W&B, both runs showed up as ``failed`` with their last steps missing.
"""

import gc
import struct
import types

import pytest

wandb = pytest.importorskip("wandb")

from omegaconf import OmegaConf  # noqa: E402

from verl.utils import tracking  # noqa: E402

# An offline run's transaction log (run-<id>.wandb) starts with a 7-byte file header, followed by a leveldb-style log
# of 32 KiB blocks. Each chunk has a 7-byte header (crc u32, length u16, type u8). Chunk types: FULL=1, FIRST=2,
# MIDDLE=3, LAST=4; a record that does not fit in the current block is split into FIRST/MIDDLE.../LAST chunks.
_BLOCK = 32768
_HEADER = 7


def _read_records(path):
    from wandb.proto import wandb_internal_pb2 as pb

    data = path.read_bytes()
    pos, pending, records = _HEADER, b"", []
    while pos < len(data):
        left = _BLOCK - pos % _BLOCK
        if left < _HEADER:  # block trailer too short for a chunk header
            pos += left
            continue
        _crc, length, kind = struct.unpack("<IHB", data[pos : pos + _HEADER])
        if kind == 0 and length == 0:  # zero padding up to the next block
            pos += left
            continue
        start, pos = pos + _HEADER, pos + _HEADER + length
        chunk = data[start:pos]
        if kind == 2:
            pending = chunk
            continue
        if kind == 3:
            pending += chunk
            continue
        record = pb.Record()
        record.ParseFromString(pending + chunk if kind == 4 else chunk)
        pending = b""
        records.append(record)
    return records


def _offline_run_records(root):
    (path,) = root.glob("wandb/offline-run-*/run-*.wandb")
    return _read_records(path)


@pytest.fixture
def offline_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    yield tmp_path
    if wandb.run is not None:  # never leave an active run behind for the next test
        wandb.finish()


def test_finish_writes_every_step_and_a_clean_exit_while_the_tracker_is_alive(offline_wandb):
    tracker = tracking.Tracking(project_name="relay-opd-test", experiment_name="finish", default_backend="wandb")
    tracker.log({"loss": 1.0}, step=1)
    tracker.log({"loss": 0.5}, step=2)

    tracker.finish()

    # `tracker` is still referenced here, so this is the explicit finish, not __del__.
    records = _offline_run_records(offline_wandb)
    kinds = [record.WhichOneof("record_type") for record in records]
    assert kinds.count("history") == 2
    assert [record.exit.exit_code for record in records if record.WhichOneof("record_type") == "exit"] == [0]


def test_collecting_a_finished_tracker_leaves_a_newer_run_running(offline_wandb):
    tracker = tracking.Tracking(project_name="relay-opd-test", experiment_name="old", default_backend="wandb")
    tracker.finish()
    newer = wandb.init(project="relay-opd-test", name="newer")

    del tracker
    gc.collect()

    assert wandb.run is newer


class _ValOnlyTrainer:
    """Stands in for the parts of RayPPOTrainer that fit() touches before its val_only return."""

    def __init__(self):
        self.config = OmegaConf.create(
            {
                "trainer": {
                    "project_name": "relay-opd-test",
                    "experiment_name": "val-only",
                    "logger": ["wandb"],
                    "val_before_train": True,
                    "val_only": True,
                },
                "skip": {},
            }
        )
        self._dump_executor = types.SimpleNamespace(_shutdown=False)
        self.checkpoint_manager = types.SimpleNamespace(update_weights=lambda global_steps: None)
        self.train_dataloader = [None]

    def _load_checkpoint(self):
        pass

    def _validate(self):
        return {"val-core/acc": 0.5}

    def _shutdown_dump_executor(self):
        pass


def test_fit_finishes_its_run_without_waiting_for_garbage_collection(offline_wandb, monkeypatch):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    alive = []

    class _KeptAlive(tracking.Tracking):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            alive.append(self)  # keep a reference, so garbage collection cannot finish the run on fit()'s behalf

    monkeypatch.setattr(tracking, "Tracking", _KeptAlive)  # fit() imports Tracking when it starts

    RayPPOTrainer.fit(_ValOnlyTrainer())

    assert len(alive) == 1
    assert wandb.run is None
