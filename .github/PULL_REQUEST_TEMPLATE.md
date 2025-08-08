# Pull Request Checklist

## Summary
- Briefly describe the change and motivation.
- Link docs or tracking (e.g., `docs/SDLC_TRACKING.md`).

## Related Issues
- Closes #<issue-id>

## Type of Change
- [ ] feat
- [ ] fix
- [ ] docs
- [ ] chore/refactor
- [ ] ci/devops

## Test Coverage
- [ ] Tests added/updated
- [ ] Ran `pytest -q`: PASS
- [ ] Tokenizer cache handled (see README Testing)

## Screenshots / Artifacts (training/eval)
- Logs/plots (loss, pass@k). Checkpoint(s): `checkpoints/...`

## Backward Compatibility
- [ ] No breaking config changes
- [ ] If breaking, include migration notes

## Quality Gates
- [ ] `black . && isort .`
- [ ] `flake8` clean
- [ ] Docs updated if user-facing behavior changed

## Operational / Security
- [ ] No secrets committed (`GITHUB_SECRETS_SETUP.md`)
- [ ] Large artifacts kept out of VCS (`data/`, `checkpoints/`, `outputs/`)

## How to Validate (copy‑paste)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
# Optional data/train/eval
python scripts/convert_mbpp.py --split all --output-dir data/mbpp --tokenizer gpt2
python scripts/train.py --config configs/hrm/mbpp_base.yaml \
  --data-path data/mbpp/train.bin --eval-data-path data/mbpp/validation.bin \
  --output-dir checkpoints/hrm-mbpp
python scripts/evaluate.py --ckpt checkpoints/hrm-mbpp/latest.pt --split test --k 1 5 10
```
