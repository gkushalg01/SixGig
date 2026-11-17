# SixGig

SixGig is a robot manipulation stack, run entirely in simulation, that does real-time AI inference and control on a laptop RTX 4050 with 6 GB of VRAM. A MacBook M4 Pro hosts the simulator and a language-model task planner. Every performance claim is backed by a reproducible benchmark file in [`results/`](results/).

> **Status:** Phase 0 is complete on the RTX laptop. The environment and benchmark harness work, and the smoke results are in [`results/phase0/`](results/phase0/). See [docs/progress.md](docs/progress.md).

## Quick start (RTX laptop)

```bash
conda env create -f env/linux-64.yml
conda activate sixgig
python -m pytest bench/tests -q   # harness unit tests
python -m bench.smoke             # Phase 0 smoke benchmarks -> results/phase0/
```

Details are in [bench/README.md](bench/README.md).

## Components

| ID | Component | Directory |
|----|-----------|-----------|
| P1 | VLA inference optimization: quantization, TensorRT, a port to MLX (Apple's ML framework) | [`vla/`](vla/) |
| P2 | Custom CUDA MPPI controller (sampling-based model-predictive control) for a 7-joint arm | [`mppi/`](mppi/) |
| P5 | Dual-machine stack: a fast controller on the RTX laptop and a slower planner on the Mac | [`system/`](system/) |
| P4 | GPU perception pipeline (stretch goal) | [`perception/`](perception/) |

VLA stands for vision-language-action model: it maps camera images and a text instruction to robot actions.

## Documentation

- [Plan](docs/plan.md): goals, architecture, phases, risks.
- [Decisions](docs/decisions.md): every decision taken, and why.
- [Glossary](docs/glossary.md): everything the project implements or depends on.
- [Progress](docs/progress.md): status and a dated log.

## Repository layout

```text
docs/         project documents
env/          conda environment files (linux-64, osx-arm64)
bench/        benchmark harness
vla/          P1
mppi/         P2
perception/   P4 (stretch goal)
system/       P5 launch files, planner, bridges
results/      benchmark JSON and plots
scripts/      helper scripts
```
