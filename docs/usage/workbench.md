# Workbench

`llm_tools.apps.textual_workbench` is a developer-facing Textual workbench for
inspecting and exercising the library interactively.

It is intentionally:

- one-turn only
- ephemeral per launch
- developer-oriented rather than end-user-facing

It does not add chat memory, replanning, or an agent loop.

## Installation

Install the optional app dependencies:

```bash
.venv/bin/python -m pip install -e .[apps]
```

## Launch

You can launch the workbench with either:

```bash
python -m llm_tools.apps.textual_workbench
```

or:

```bash
llm-tools-workbench
```

## What It Can Do

The workbench exposes four main activities:

- inspect the currently registered built-in tools
- export adapter-specific tool descriptions
- run one provider-backed model turn through any supported mode
- execute one tool directly through `ToolRuntime`

## Default Behavior

On startup the workbench:

- enables filesystem, git, and text tools
- leaves Atlassian tools disabled by default
- uses an Ollama-oriented provider preset by default
- keeps all configuration in memory only

## Interaction Modes

The workbench supports:

- native tool calling
- structured output
- prompt schema

For provider-backed turns it uses `OpenAICompatibleProvider`, then passes the
parsed result into `WorkflowExecutor` when execution-after-parse is enabled.
Provider and workflow calls use async execution paths under the hood while
preserving the same visible one-turn behavior.

## Direct Tool Execution

The direct tool section lets you:

- choose a registered tool name
- provide JSON arguments
- run the tool without calling any provider

This is useful for inspecting runtime normalization, policy decisions,
observability, and execution records even when no model endpoint is available.
