# GPUSLS-LEO solver core

This checkout provides the JAX SQP, SLS and ADMM solvers used by the parent
project. Callers supply dynamics, costs, constraints, disturbances and
linearization-error bounds. The parent project supplies the contact-specific
adapters and certificates.

From the parent repository, initialize GPUSLS-LEO and its pinned iLQR dependency:

```bash
git submodule update --init gpusls-leo
git -C gpusls-leo submodule update --init src/gpu_sls/external/primal_dual_ilqr
```

Install the core into the active Python environment:

```bash
python -m pip install -e ./gpusls-leo
```

For an environment with the solver dependencies already installed, adding
`gpusls-leo/src` to `PYTHONPATH` also works. Choose the JAX accelerator
installation for your environment if running on a GPU.
