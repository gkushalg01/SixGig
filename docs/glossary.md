# Glossary

Terms for everything this project implements or depends on. Each entry says what it is and where it appears in the project. Phases refer to [plan.md](./plan.md).

## Architecture

- **SixGig:** the project name. It refers to building real-time robot intelligence inside the RTX 4050's 6 GB of VRAM.
- **System 1 / System 2:** a split between a fast, reactive controller (System 1: the VLA and MPPI on the RTX laptop) and a slow, deliberative planner (System 2: the VLM on the Mac). Used in Phase 3.
- **Dual-machine stack:** the Mac and RTX laptop working as one robot brain over wired LAN. See D-011 in [decisions.md](./decisions.md).
- **Latency budget:** a per-stage breakdown of end-to-end delay: render, transport, perception, VLA, MPPI, actuation. Produced in Phase 3.

## Models and learning

- **VLA (vision-language-action model):** a model that maps camera images and a language instruction to robot actions. Optimized in Phase 1.
- **SmolVLA:** a small open VLA of about 0.5B parameters, supported by lerobot. It is the primary candidate under D-004.
- **VLM (vision-language model):** a model that reasons over images and text. Used as the task planner on the Mac in Phase 3.
- **Action chunking:** predicting a short sequence of future actions in one inference instead of one action per step.
- **Asynchronous action chunking:** computing the next chunk while the robot executes the current one, which hides inference latency. Phase 1.
- **Flow matching / denoising steps:** the iterative refinement some VLA action heads use to produce actions. Fewer steps means lower latency but possibly lower accuracy. Phase 1.
- **Prefix caching (KV cache reuse):** reusing attention keys and values for the unchanged prompt and image tokens across inference calls. Phase 1.
- **Constrained decoding:** forcing a language model's output to follow a grammar or schema, such as JSON subtasks, so the robot can parse it. Phase 3.
- **Imitation learning:** training a policy from demonstrations. The VLAs used here are pretrained this way.
- **Open-vocabulary detection:** detecting objects named in free text rather than from a fixed class list. Stretch goal P4.

## Inference optimization

- **Quantization:** storing weights or activations in fewer bits to save memory and time. This project uses INT8, and 4-bit as NF4 or AWQ. Phase 1.
- **FP16 / BF16:** 16-bit floating-point formats, the usual baseline for GPU inference.
- **ONNX:** a portable model format, used as the export path into TensorRT.
- **TensorRT:** NVIDIA's inference compiler. It turns a model into an optimized engine with fused kernels and tuned precision. Phase 1 and P4. See D-009.
- **TensorRT engine:** the compiled, GPU-specific output of TensorRT. It is tied to the GPU model and TensorRT version.
- **`torch.compile`:** PyTorch's just-in-time compiler. It is the fallback when TensorRT export fails.
- **MLX:** Apple's array framework for Apple-silicon unified memory. Used for the VLA port and the VLM planner on the Mac.
- **Pareto frontier:** the set of configurations where no other configuration is better on every axis at once. Here the axes are task success, latency and VRAM. It is the headline chart of Phase 1.

## Control and planning

- **MPPI (Model Predictive Path Integral control):** a sampling-based controller. Each step it simulates thousands of perturbed action sequences, weights them by cost, and averages them into the next command. Built in CUDA in Phase 2.
- **Rollout:** simulating one candidate action sequence forward through a dynamics model.
- **Batched rollouts:** running thousands of rollouts in parallel, one or more per GPU thread. This is the core of MPPI's GPU speedup.
- **Cost kernel:** the CUDA kernel that scores each rollout, for example by goal distance, collisions, joint limits and smoothness.
- **Sampling prior:** the distribution MPPI samples around. In Phase 3 it is centred on the VLA's action chunk, so the learned policy proposes and the optimizer refines.
- **Forward kinematics:** computing link poses from joint angles. It is needed inside the rollout kernel for a 7-DoF arm.
- **7-DoF arm:** a manipulator with seven joints, such as the Franka Panda used in LIBERO.

## GPU and HPC

- **CUDA kernel:** a function that runs in parallel across many GPU threads.
- **sm_89:** the compute capability of the Ada Lovelace generation, which includes the RTX 4050. All custom kernels target it.
- **Kernel fusion:** merging several kernels into one to avoid round trips through GPU memory.
- **Memory coalescing / SoA layout:** arranging data as a structure of arrays so neighbouring threads read neighbouring addresses.
- **CUDA graphs:** recording a fixed sequence of kernel launches once and replaying it, which removes per-launch CPU overhead. Phase 2.
- **CUDA streams:** independent queues of GPU work that can overlap, for example copying data during compute.
- **Pinned memory:** host memory locked in RAM, which allows fast, asynchronous transfers to the GPU.
- **Nsight Systems / Nsight Compute:** NVIDIA profilers for the whole timeline and for individual kernels.
- **Roofline analysis:** comparing a kernel's achieved throughput with the hardware's memory and compute limits to see which one bounds it.
- **VRAM budget:** an explicit allocation of the 6 GB across models, buffers and CUDA context. It is enforced by the P4 scheduler.

## Simulation and robotics middleware

- **MuJoCo:** the physics simulator used for all tasks. See D-003.
- **LIBERO:** a MuJoCo-based suite of language-conditioned manipulation tasks, used to measure task success.
- **ROS 2:** robot middleware for nodes, topics and services. Version Jazzy, see D-010.
- **RoboStack:** conda-forge packages of ROS 2 that work on Linux and macOS.
- **Composable nodes / intra-process communication:** running several ROS 2 nodes in one process so messages pass by pointer instead of being copied. Used for zero-copy in P4.
- **Zero-copy:** passing data between stages without copying it, ideally keeping it on the GPU throughout.
- **DDS / Zenoh / rmw_zenoh:** ROS 2 transport layers. Zenoh suits links between machines.

## Measurement

- **Benchmark harness:** the shared tool that runs workloads and records the metrics required by D-012. Built in Phase 0.
- **p50 / p95 / p99:** latency percentiles. p99 shows the worst-case behaviour that matters for control.
- **Thermal throttling:** a laptop GPU lowering its clocks when hot, which distorts benchmarks unless recorded.
- **Task success rate:** the fraction of seeded episodes in which the robot completes the task.

## Environment

- **Secure Boot / MOK (Machine Owner Key):** UEFI module signing. With Secure Boot on, the NVIDIA module loads only if its signing key is enrolled. The key is enrolled, but Secure Boot is currently off. See D-005 and D-014.
- **DKMS:** rebuilds the NVIDIA kernel module, and re-signs it, when the kernel updates.
- **conda environment `sixgig`:** the isolated Python 3.12 environment for the project. See D-007.
- **`i915` / `xe`:** the two Intel graphics kernel drivers. `i915` is the mature one and drives the laptop's Alder Lake iGPU (PCI ID 46a3). `xe` is the newer one and is blacklisted because it froze this machine. See D-015.
- **Kernel command-line parameters:** boot options set in `GRUB_CMDLINE_LINUX_DEFAULT`, here `i915.force_probe=46a3 modprobe.blacklist=xe`. See D-015.
- **PRIME on-demand:** the hybrid-graphics mode in which the Intel iGPU drives the display and the NVIDIA GPU runs only the work sent to it, such as CUDA.
