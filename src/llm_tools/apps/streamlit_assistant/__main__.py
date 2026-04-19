"""Module entrypoint for the Streamlit assistant app."""


def _main() -> int:
    from llm_tools.apps.streamlit_assistant import main

    return main()


def main() -> int:
    """Return the console entrypoint result."""
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
