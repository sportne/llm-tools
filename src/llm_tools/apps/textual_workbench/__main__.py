"""Module entrypoint for the Textual workbench."""


def _main() -> int:
    from llm_tools.apps.textual_workbench import main

    return main()


def main() -> int:
    """Return the console entrypoint result."""
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
