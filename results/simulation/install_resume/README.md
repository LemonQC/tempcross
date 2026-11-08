# Verified installation evidence

These files record the independent BARX runtime check: pinned versions, CUDA,
FlashAttention kernel, NVIDIA EGL, robot registrations and 76 upstream tests.
`runtime_verification.json` intentionally records zero manipulation episodes because
that command does not run a task. Closed-loop evidence is in `../closed_loop/`.

`dependency_check.txt` lists four metadata conflicts inherited from upstream pins;
it is not a clean pip-check claim. `mirror-manifest.json` retains official registry
artifact hashes and fixed Git commits. FlashAttention's SHA is recorded locally for
the official release wheel, not claimed as an upstream published checksum. Model
integrity records distinguish published manifest hashes from pinned HF LFS/cache hashes.
