<img width="1271" height="790" alt="Screenshot 2026-08-13 at 17 28 46" src="https://github.com/user-attachments/assets/8a13725e-85d7-47f7-8cb4-a7a6d2d1fd9a" />

Streamlit app using the Ollama Python SDK.

## Setup

Use the `agent-interface` conda environment:

```bash
conda activate agent-interface
streamlit run app.py
```

## Running

`app.py` is the main chat interface.

Configuration lives in the `ChatConfig` dataclass, which exposes `to_ollama_options()` for the Ollama API payload.

The sidebar UI is isolated in `render_sidebar()`, message history is rendered by `render_messages()`, and streaming is split into `stream_with_thinking()` and `stream_without_thinking()` depending on the `think` toggle.

The model name is declared once as `MODEL`.

For Ollama Python library references, see `ollama-python-ref/`.
