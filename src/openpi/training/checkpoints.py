from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import shutil
import subprocess
import threading
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    keep_all: bool = False,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            # keep_all is used when an external uploader (S3CheckpointSync) owns
            # local-disk pruning: orbax must not garbage-collect checkpoints out
            # from under an in-flight upload, so it keeps every checkpoint and
            # the uploader is the sole deleter.
            max_to_keep=None if keep_all else 1,
            keep_period=None if keep_all else keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


class S3CheckpointSync:
    """Streams checkpoints to S3 in the background, keeping only the newest on disk.

    After each save, every checkpoint older than the latest is uploaded to S3 in
    a background thread; once an upload succeeds the local copy is deleted. The
    newest checkpoint is always kept on local disk so training can resume or be
    evaluated from it. A checkpoint is never deleted locally before its upload
    has completed successfully -- a failed upload is logged and retried on the
    next save. Checkpoints land at ``s3://<bucket>/<prefix>/<step>/``.

    Use together with ``initialize_checkpoint_dir(..., keep_all=True)`` so orbax
    does not garbage-collect a checkpoint while its upload is in flight.
    """

    def __init__(self, *, checkpoint_dir: epath.Path | str, bucket: str, prefix: str):
        self._checkpoint_dir = epath.Path(checkpoint_dir)
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        # Single worker: uploads run sequentially, bounding disk usage and
        # network contention while still being off the training thread.
        self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="s3-ckpt")
        self._lock = threading.Lock()
        self._inflight: set[int] = set()
        self._uploaded: set[int] = set()

    def _s3_uri(self, step: int) -> str:
        return f"s3://{self._bucket}/{self._prefix}/{step}"

    def _local_steps(self) -> list[int]:
        """Committed checkpoint steps currently on local disk, sorted ascending."""
        steps = []
        for path in self._checkpoint_dir.iterdir():
            # orbax names a fully committed checkpoint with the bare step int;
            # an in-progress one carries a ``.orbax-checkpoint-tmp-*`` suffix.
            if path.is_dir() and path.name.isdigit():
                steps.append(int(path.name))
        return sorted(steps)

    def after_save(self, latest_step: int) -> None:
        """Queue a background upload+prune for every checkpoint older than latest_step."""
        for step in self._local_steps():
            if step >= latest_step:
                continue  # never touch the newest checkpoint -- it stays on disk
            with self._lock:
                if step in self._inflight or step in self._uploaded:
                    continue
                self._inflight.add(step)
            self._executor.submit(self._upload, step, delete=True)

    def upload_final(self, step: int) -> None:
        """Upload the newest checkpoint to S3 synchronously, keeping it on disk."""
        with self._lock:
            if step in self._uploaded or step in self._inflight:
                return
            self._inflight.add(step)
        self._upload(step, delete=False)

    def _upload(self, step: int, *, delete: bool) -> None:
        local = self._checkpoint_dir / str(step)
        try:
            if not local.exists():
                logging.warning(f"[s3-ckpt] checkpoint {step} not on disk; skipping upload")
                return
            uri = self._s3_uri(step)
            logging.info(f"[s3-ckpt] uploading checkpoint {step} -> {uri}")
            subprocess.run(
                ["aws", "s3", "sync", "--no-progress", str(local), uri],
                check=True,
                capture_output=True,
                text=True,
            )
            with self._lock:
                self._uploaded.add(step)
            logging.info(f"[s3-ckpt] upload of checkpoint {step} complete")
            if delete:
                # Delete locally only after a verified successful upload.
                shutil.rmtree(str(local))
                logging.info(f"[s3-ckpt] pruned local checkpoint {step}")
        except subprocess.CalledProcessError as exc:
            logging.error(f"[s3-ckpt] upload of checkpoint {step} failed (will retry): {exc.stderr}")
        except Exception as exc:  # noqa: BLE001
            logging.error(f"[s3-ckpt] error handling checkpoint {step}: {exc!r}")
        finally:
            with self._lock:
                self._inflight.discard(step)

    def wait(self) -> None:
        """Block until all queued uploads finish (call once at end of training)."""
        self._executor.shutdown(wait=True)
