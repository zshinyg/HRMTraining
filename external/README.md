# External Dependencies

This directory is reserved **exclusively for third-party repositories and vendor code** that our project depends on but does not maintain.

## Sapient HRM (primary dependency)

During the *HRM → CodeGen* adaptation effort we will clone the official open-source **Hierarchical Reasoning Model** from Sapient Inc. (Apache-2.0 licence, GitHub ★6.4 k):

```
git clone https://github.com/sapientinc/HRM.git external/sapient-hrm
# ‑- OR add as a submodule
git submodule add https://github.com/sapientinc/HRM.git external/sapient-hrm
```

Once cloned, the directory structure will look like:

```
external/
└── sapient-hrm/     # upstream code, unmodified
```

We keep the upstream code **unaltered**; all adaptation layers live in our own source tree.  
If fixes or changes are required, contribute them upstream or patch via a separate directory, never commit them directly inside `external/sapient-hrm`.

## Git & CI notes

* Large checkpoints and build artefacts should be excluded via `.gitignore`.
* CI steps that need HRM must run `git submodule update --init --recursive` or clone on-demand.
* Review third-party licences before release; Sapient HRM is Apache-2.0 compatible with our project.

---
_Last updated: 2025-08-05_
