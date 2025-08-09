---
status: active
owner: security
summary: Legal code ownership verification checklist and process
last_reviewed: 2025-08-08
---

# LEGAL_CODE_OWNERSHIP_VERIFICATION.md
_HRM-CodeGen Project_  
_Last updated: 2025-08-05_

---

## 1 · Repository Ownership Verification

| Item | Evidence | Verdict |
|------|----------|---------|
| **GitHub repository** | `https://github.com/zshinyg/HRMTraining` – owner shown as **zshinyg** | ✅ Sole owner |
| **Default branch** | `main` protected, maintained by zshinyg | ✅ |
| **Commit signatures** | All commits authored under *zshinyg* account (Droids commit via author-spoof header set to zshinyg) | ✅ |
| **CLA / ToS** | GitHub ToS §B grants repository owner full license to contributions | ✅ |
| **Employment / Work-for-hire** | All Droids act as tools under user direction ≈ work-for-hire; no independent IP claims | ✅ |

**Conclusion:** zshinyg holds **exclusive ownership** of the repository contents.

---

## 2 · File Contribution & Attribution Log

| Path | Primary Author (Git `--follow`) | Notes |
|------|---------------------------------|-------|
| `/hrm_codegen/**` | zshinyg (SE Code Droid commits) | Original implementation |
| `/datasets/**` | zshinyg | Data loaders written in-house |
| `/scripts/**` | zshinyg (Infra / Reliability Droids) | Infrastructure & monitoring |
| `/configs/**` | zshinyg | Model & training configs |
| `/documentation *.md` | zshinyg (Product Droid) | All markdown authored by owner |
| `/external/sapient-hrm` | **Upstream Sapient Inc.** (Submodule, untouched) | Apache-2.0 licensed, attribution retained |

_No third-party code committed inside project tree except under `external/`._

---

## 3 · Third-Party Dependencies & Licenses

| Dependency | Version / Commit | License | Included Via |
|------------|------------------|---------|--------------|
| **Sapient HRM** | `14f3c5a` | Apache-2.0 | Git submodule (`external/`) |
| **PyTorch** | 2.2.0 | BSD-3-Clause | PyPI |
| **Transformers** | 4.41.0 | Apache-2.0 | PyPI |
| **Datasets** (`huggingface`) | 2.19 | Apache-2.0 | PyPI |
| **tqdm** | 4.66 | MPL-2.0 | PyPI |
| **numpy** | 1.26 | BSD-3-Clause | PyPI |
| **scipy** | 1.13 | BSD-3-Clause | PyPI |

All other transitive deps inherit permissive licenses (MIT/BSD/Apache).

---

## 4 · Open-Source Compliance Checklist

| Check | Status | Notes |
|-------|--------|-------|
| License file present (`LICENSE`) | ✅ MIT (project choice) |
| NOTICE file for Apache code | 🔄 To create (see §8) |
| Submodule kept unmodified | ✅ |
| Package licenses documented | ✅ §3 table |
| No GPL/AGPL code imported | ✅ Verified via `pip-licenses` scan |
| Source/header license headers | n/a – permissive licenses do not require per-file headers |

---

## 5 · Code Attribution Verification for Droids

All Droids act under the **Factory Terms of Service** as tools, not legal persons.  
Their outputs are contributed under the **repository owner account (zshinyg)**; therefore, copyright and related rights in those contributions vest in **zshinyg**.

_No additional attribution obligations_ beyond normal OSS NOTICE requirements.

---

## 6 · License Compatibility Analysis

| Component | License | Compatible with MIT Project? | Rationale |
|-----------|---------|------------------------------|-----------|
| Sapient HRM | Apache-2.0 | ✅ | Notice preservation required |
| PyTorch | BSD-3 | ✅ | Permissive |
| Transformers | Apache-2.0 | ✅ | Permissive, notice file |
| tqdm / numpy / scipy | MPL-2.0 / BSD-3 | ✅ | MPL-2.0 allows MIT aggregation |

No copyleft (GPL/AGPL) inbound; distribution under MIT is legally sound.

---

## 7 · Legal Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Missing NOTICE for Apache deps | Med | Low-Med | Add consolidated NOTICE file |
| Submodule accidental modification | Low | Low | Enforce read-only CI check |
| Patent clauses (Apache-2.0) | Low | Low | Acceptable under Apache patent grant |
| Dataset licensing (MBPP) | Low | Med | MBPP is MIT – compliant |
| Confidential info in commits | Very Low | High | Repository audit – no sensitive data present |

Overall **legal risk: Low** once NOTICE file added.

---

## 8 · Recommended Actions

1. **Create `NOTICE` file** summarising Apache-2.0 attributions  
   • Sapient HRM, HuggingFace Transformers  
2. **Add “Third-Party Licenses” section to README** linking §3 table  
3. **Run automated license scan in CI** (`pip-licenses --format=markdown`)  
4. **Lock `external/sapient-hrm` submodule to commit hash**; forbid edits via CI  
5. **Maintain contributors file** (`CODEOWNERS` optional) noting zshinyg sole owner  
6. **Keep project LICENSE = MIT** (no inbound conflicts)  
7. **Periodic audits** (quarterly) for new dependencies or license changes  
8. **Document work-for-hire status** in CONTRIBUTING.md to cover future collaborators  

---

**Conclusion:** After implementing the NOTICE file and CI license scan, zshinyg will hold **clear, uncontested legal ownership** of all project code with full open-source compliance.
