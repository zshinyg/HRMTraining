# GITHUB_SECRETS_SETUP.md  
_HRM-CodeGen · Repository Secret Configuration Guide_  
_Last updated: 2025-08-05_

---

## 1 · Required Secrets & How to Obtain Them

| Secret Key | Used By | Where to Obtain | Scope / Least-Privilege Notes |
|------------|---------|-----------------|------------------------------|
| `WANDB_API_KEY` | CI `benchmark.yml`, `train.py` | Weights & Biases → Settings → API Keys | Create **project-specific** key (not personal) with “Artifacts: read/write” only |
| `DOCKER_USERNAME` | CI `ci.yml`, `benchmark.yml` | Docker Hub account | Service account with **write** access to targeted repo |
| `DOCKER_PASSWORD` | ↑ | Docker Hub → “Access Tokens” | Generate **access token**, never store plaintext password |
| `HF_TOKEN` | Optional model upload | Hugging Face → Settings → Access Tokens | **Write** scope for model repo, read for datasets |
| `SLACK_WEBHOOK` (optional) | `monitoring.yml` alerts | Slack App → Incoming Webhooks | Create dedicated #ci-alerts channel |
| `BASELINE_MODEL_PATH` | `benchmark.yml` | Internal S3 / HF | Path/URL pointer – no secrets if public |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (optional) | Trainer logs → S3 | AWS IAM console | IAM user with **PutObject** to specific bucket |
| `GPG_PRIVATE_KEY` (optional) | Release signing | Local GPG | Export armored key, protect with passphrase env var `GPG_PASSPHRASE` |

> 🛡️ **Ownership:** All secrets are owned by **zshinyg**. Do **NOT** commit them to git.

---

## 2 · Adding Secrets via GitHub UI (Step-by-Step)

1. **Open Repository** → `Settings` tab  
2. Left sidebar → **Secrets and variables** → **Actions**  
3. Click **“New repository secret”**  
4. **Name**: exact key (e.g., `WANDB_API_KEY`)  
5. **Value**: paste secret token / key  
6. Click **Add secret** ✅  
7. Repeat for each required key.

> ⚠️ **Tip:** Repository secrets are **encrypted at rest** and **masked in logs** (≥ 4 chars, no `*` or `=` contiguous).

---

## 3 · Environment-Specific Configuration

| Environment | Secret Naming Convention | Example |
|-------------|--------------------------|---------|
| **dev** | `<KEY>_DEV` | `WANDB_API_KEY_DEV` |
| **staging** | `<KEY>_STG` | `DOCKER_PASSWORD_STG` |
| **prod** | `<KEY>_PROD` | `HF_TOKEN_PROD` |

Implement via **Environment Secrets**:

1. Settings → **Environments** → `New environment` (`dev`, `staging`, `prod`)  
2. Add secrets inside each environment.  
3. In workflow, reference with `secrets.WANDB_API_KEY_PROD` etc.  
4. Protect prod environment with required reviewers (`CODEOWNERS`).

---

## 4 · Security Best Practices

1. **Least privilege:** Scope tokens to single project/repo.  
2. **Rotation:** Rotate secrets every 90 days; document in `SECURITY_FRAMEWORK.md`.  
3. **No plaintext output:** Never `echo $SECRET`; use `***` masking or pass to tools directly.  
4. **Branch protection:** Require PR reviews before workflow changes.  
5. **Audit:** Enable GitHub audit log; review secret usage monthly.  
6. **Separate service accounts:** Avoid personal accounts for CI.  
7. **Automated scanning:** Dependabot + `security.yml` run Trivy on images.  

---

## 5 · Validation Steps

| Check | Command / Action | Expected Result |
|-------|------------------|-----------------|
| CI redaction | `run: echo $WANDB_API_KEY` in PR | Log shows `***` |
| Docker login | `docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD` | `Login Succeeded` |
| W&B ping | `wandb login --relogin $WANDB_API_KEY` | “Successfully logged in” |
| HF upload (opt) | `python scripts/hf_check.py` | “Authorized” |
| Benchmark CI | Push dummy PR → watch `benchmark.yml` | Jobs pass, W&B run appears |
| Slack alert (opt) | Trigger workflow failure | Slack message delivered |

---

## 6 · Troubleshooting Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `***` printed instead of secret value | GitHub masks secrets ≥4 chars | Normal; use tool that consumes var |
| `docker login: denied` | Wrong token scope | Generate new access token with write perms |
| `wandb: Permission denied` | Invalid / expired key | Regenerate key in W&B → re-add secret |
| CI step `not permitted to access` | Secret defined as env, not repo secret | Re-add under **Actions secrets** |
| Prod workflow fails, dev works | Environment protection rules block prod | Add reviewers or use `dev` env secrets |
| Flash-Attn compile workflow cannot read key | Secret name mismatch | Confirm `secrets.HF_TOKEN` matches workflow |

---

## 7 · Integration Notes

### 7.1  Weights & Biases (W&B)
```yaml
env:
  WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  WANDB_PROJECT: hrm-validation
```
- CI publishes metrics & plots.
- Dashboard links auto-posted via Slack webhook (optional).

### 7.2  Docker Hub
```yaml
steps:
  - name: Login to Docker Hub
    run: echo "${{ secrets.DOCKER_PASSWORD }}" | \
         docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
```

### 7.3  Hugging Face (optional)
```yaml
env:
  HF_TOKEN: ${{ secrets.HF_TOKEN }}
```
- Used for pushing model checkpoints or pulling gated datasets.

### 7.4  AWS / S3 (optional)
```yaml
env:
  AWS_ACCESS_KEY_ID:     ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```
- Restrict IAM user to project bucket only.

---

## 8 · Change Log

| Date | Version | Change |
|------|---------|--------|
| 2025-08-05 | 1.0 | Initial secret configuration guide |

---

_Questions? Ping **Product Droid** in Factory thread._  
