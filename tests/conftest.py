"""Reviewer-facing pytest options and failure guidance."""


def pytest_addoption(parser):
    parser.addoption(
        "--check-live",
        action="store_true",
        default=False,
        help="Validate local .env settings without contacting a provider.",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if exitstatus == 0:
        terminalreporter.write_line(
            "Offline artifact checks passed; no model requests were made."
        )
        if config.getoption("--check-live"):
            terminalreporter.write_line(
                "Local live settings passed; provider access, credit and current model availability are not tested."
            )
    else:
        terminalreporter.write_line(
            "See README.md: Installation, Sanity tests, and Troubleshooting."
        )
        terminalreporter.write_line(
            "Activate .venv and install the documented packages before rerunning uv run --no-project python -m pytest -q."
        )
