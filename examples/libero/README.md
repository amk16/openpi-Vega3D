# LIBERO Eval

Evaluate any pi05 checkpoint on LIBERO and LIBERO-Pro benchmarks.

## Setup

Handled by `vastai/Dockerfile` — Python 3.8 venv at `/venv/libero`, separate from the main py3.11 env. LIBERO-Pro data files download automatically on first boot via `vastai/onstart.sh`.

## Running

```bash
# Terminal 1 — model server
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir <checkpoint_path>

# Terminal 2 — eval
bash scripts/run_libero_eval.sh \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 10 \
  --args.video-out-path data/libero/videos
```

## Output

- Results: `data/libero/results/<suite>_results.json`
- Videos: `data/libero/videos/<suite>/`

## Suites

| Type | Suites |
|------|--------|
| Standard | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| Pro | `libero_{spatial,object,goal,10}_{object,swap,lan,task,env,temp}` |

## Reference Results

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| pi0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
