# Prompts

Every prompt the agent uses lives here as a versioned Markdown file — **never as
an inline string in Python**. Naming: `<name>_v<N>.md`. A prompt change is a new
version (new file), never an edit to an old one, so the journal's history stays
interpretable: any past iteration can be traced to the exact prompt text it ran
with (record the prompt name in the iteration `config`).

Referenced by the `PROMPT_NAME` constants in `agent/actions/*.py` and by
`agent/policy.py` / `agent/critic.py`, loaded through `LLMClient.load_prompt`.

Template variables use `{variable_name}` placeholders filled from the
`build_context` dict of the owning action.

Current files are v1 placeholders — structure only, content to be written when
the harness milestone is done.
