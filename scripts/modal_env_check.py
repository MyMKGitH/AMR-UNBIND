"""Manual diagnostic: is this environment ready to dispatch AMR-UNBIND runs to Modal?

NOT part of the automated test suite (tests/test_cloud.py covers what pytest
runs). This is a standalone script you run by hand when a Modal cloud run
isn't working and you want to know which of three things is missing, before
digging further:

    python scripts/modal_env_check.py

Checks, in order: (1) whether Modal-related environment variables are
present, (2) whether `amr_unbind` and the `modal` package both import
cleanly, and (3) whether `amr_unbind/cloud.py` has actually been deployed
(`modal deploy amr_unbind/cloud.py`) and is reachable. Each step reports
plainly and the script continues past a failure so you see the full
picture in one run, rather than stopping at the first problem.

This performs no GPU dispatch and should not incur any cost. For an actual
live Modal GPU smoke test, see modal_live_dispatch_check.py instead --
that one does dispatch a real (tiny, cheap) job and requires a working
Modal account.
"""

import os
import sys
import traceback


def check_cloud_dispatch() -> None:
    print("=== AMR-UNBIND Modal Environment Check ===")

    # 1. Inspect environment variables for cloud credentials.
    print("\n[1/3] Checking environment variables...")
    cloud_keys = [
        k
        for k in os.environ
        if any(term in k.upper() for term in ["MODAL_TOKEN", "MODAL_"])
    ]
    if cloud_keys:
        print("  Found Modal-related environment variables:")
        for key in cloud_keys:
            val = os.environ[key]
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
            print(f"    - {key}: {masked}")
    else:
        print(
            "  No MODAL_* environment variables detected in this session. "
            "That's normal if you authenticated via `modal token new` "
            "(it writes to ~/.modal.toml, not the environment) -- this "
            "check is informational, not a failure by itself."
        )

    # 2. Import amr_unbind and the modal package.
    print("\n[2/3] Checking imports...")
    try:
        import amr_unbind  # noqa: F401

        print(f"  [PASS] Imported amr_unbind (version {amr_unbind.__version__}).")
    except ImportError as exc:
        print(f"  [FAIL] Could not import amr_unbind: {exc}")
        print("         Run this script from the project root, with the project installed")
        print("         (`pip install -e .`) or on PYTHONPATH.")
        return

    try:
        from amr_unbind.cloud import modal_available

        if not modal_available():
            print(
                "  [FAIL] The `modal` package is not installed in this environment. "
                "Run `pip install -r requirements-cloud.txt`."
            )
            return
        print("  [PASS] `modal` package is installed.")
    except ImportError as exc:
        print(f"  [FAIL] Could not import amr_unbind.cloud: {exc}")
        return

    # 3. Check whether amr_unbind/cloud.py has actually been deployed and is
    #    reachable -- the single most common reason a real run stalls or
    #    fails is that `modal deploy amr_unbind/cloud.py` was never run.
    print("\n[3/3] Checking whether amr_unbind/cloud.py is deployed to Modal...")
    try:
        from amr_unbind.cloud import MODAL_APP_NAME, MODAL_FUNCTION_NAME, _lookup_modal_function

        fn = _lookup_modal_function()
        print(f"  [PASS] Found deployed function '{MODAL_FUNCTION_NAME}' in app '{MODAL_APP_NAME}'.")
        print("         Cloud runs should be able to dispatch. For an actual live GPU")
        print("         smoke test, run scripts/modal_live_dispatch_check.py.")
    except Exception as exc:
        print(
            f"  [FAIL] Could not find a deployed function '{MODAL_FUNCTION_NAME}' in "
            f"app '{MODAL_APP_NAME}'."
        )
        print("         Run `modal deploy amr_unbind/cloud.py` once, then re-run this check.")
        print("\n  Full error, for reference:")
        print("  " + "=" * 58)
        traceback.print_exc(file=sys.stdout)
        print("  " + "=" * 58)


if __name__ == "__main__":
    check_cloud_dispatch()
