# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository overview

- Project name: `FYP` (from `README.md`).
- As of now, the repository only contains `README.md` and no source code, configuration files, or tests.
- There is no established application architecture yet; future contributors should document major modules and subsystems here as they are introduced.

## Commands and tooling

At this stage, there are no language-specific build, lint, or test commands defined in this repository (no `package.json`, `pyproject.toml`, `Makefile`, solution files, or similar build/test configuration are present).

When build and test tooling is added, update this section with concrete commands. For example:
- **Build:** document the primary build command (e.g., `npm run build`, `mvn package`, `dotnet build`, etc.).
- **Run all tests:** document the main test command (e.g., `npm test`, `pytest`, `dotnet test`, etc.).
- **Run a single test:** show how to scope tests to a single file or test case (e.g., `pytest path/to/test_file.py::TestClass::test_case`).
- **Lint/format:** add any linting/formatting commands once configured.

## Guidance for future updates to this file

As the project evolves, keep this file in sync with the actual repository state:
- Describe the high-level architecture once there are multiple modules (e.g., list core services, API layers, front-end apps, shared libraries, and how they interact).
- Add sections for environment setup only if there are non-trivial steps (e.g., required services, databases, or build tools).
- Keep command examples accurate and specific to the tooling actually used in this repo.
