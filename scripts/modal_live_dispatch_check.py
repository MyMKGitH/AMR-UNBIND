"""Manual diagnostic: can this machine actually reach a Modal GPU right now?

    python scripts/modal_live_dispatch_check.py

*** This dispatches a real, live job to Modal and will incur real cost on
*** your account (a tiny one -- a few seconds on the cheapest available
*** GPU -- but not zero). Only run this when you specifically want to
*** confirm Modal connectivity/authentication/GPU access work at all,
*** separately from whether amr_unbind/cloud.py itself is deployed
*** correctly (see modal_env_check.py for that, which costs nothing).

Deliberately uses a tiny, throwaway image (just PyTorch) rather than this
project's own multi-gigabyte Dockerfile.docking-based image, so a basic
connectivity check doesn't also require building the full scientific
image first. It also deliberately uses the *ephemeral* `app.run()`
pattern (not `Function.from_name`), since the point here is to test raw
Modal connectivity, independent of whether AMR-UNBIND's own function has
been deployed yet -- that's a separate thing modal_env_check.py verifies.
"""

import os
import sys
import traceback

import modal

os.environ["PYTHONUNBUFFERED"] = "1"

image = modal.Image.debian_slim().pip_install("torch")
app = modal.App("amr-unbind-debug")


@app.function(image=image, gpu="any")
def remote_ping():
    import torch

    return f"Remote GPU active: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}"


def run_check() -> None:
    print("=== AMR-UNBIND Live Modal Dispatch Check ===")
    print("This will dispatch a real job to Modal and may incur a small cost.")
    print("Triggering remote Modal GPU execution...\n")
    try:
        with app.run():
            result = remote_ping.remote()
            print(f"   [PASS] {result}\n")
            print(
                "Modal connectivity, authentication, and GPU access all work. If a "
                "real AMR-UNBIND cloud run still fails, run scripts/modal_env_check.py "
                "next -- the most common remaining cause is that amr_unbind/cloud.py "
                "hasn't been deployed yet (`modal deploy amr_unbind/cloud.py`)."
            )
    except Exception:
        print("\n[FAIL] Remote Modal dispatch failed with an exception:")
        print("=" * 60)
        traceback.print_exc(file=sys.stdout)
        print("=" * 60)
        print(
            "\nIf this mentions authentication, run `modal token new`. If it "
            "mentions GPU availability, try again shortly -- GPU pools can be "
            "temporarily full."
        )


if __name__ == "__main__":
    run_check()
