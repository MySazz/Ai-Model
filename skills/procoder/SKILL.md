---
name: procoder
description: >-
  Work like a senior developer in a repository governed by procoder: run the
  commit gate before calling anything done, format and lint through the
  binary, and drive the spec, plan, todo, backlog, and sprint chain in
  .procoder/. Use this skill when the repository contains a .procoder/
  directory or an AGENTS.md naming procoder, or when the user asks to run the
  gate, check formatting, open a spec or plan, close a task, or prepare a
  release.
license: Apache-2.0
metadata:
  category: development
  author: pascal-watteel
---

# Repository Instructions

## Project boundary

The complete Hybrid Agent project must live inside this repository root.

Keep all source code, tests, documentation, configuration, scripts, assets,
development tooling, and reproducible setup instructions within this directory
so the repository can later be uploaded to GitHub as a self-contained project.

Do not create required project components in sibling repositories or personal
directories. Temporary, disposable test artifacts may use `/tmp`, but the
project must not depend on them.

Keep credentials and runtime secrets out of version control. Document required
environment variables in the repository and provide safe example configuration
files when configuration is introduced.
