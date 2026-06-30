import json
import os
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Union

import streamlit as st
import ollama
from ollama import Client, WebSearchResponse, WebFetchResponse
from markitdown import MarkItDown

MODEL = "minimax-m3:cloud"
TOOL_RESULT_LIMIT = 4000000

with open(os.path.join(os.path.dirname(__file__), "prompts", "default-system-prompt-1.md"), "r") as _f:
    SYSTEM_PROMPT = _f.read()
TIMESTAMP_FORMAT = "<system_time>%A %Y-%m-%d %H:%M:%S %Z</system_time>"
CHATS_DIR = os.path.join(os.path.dirname(__file__), "chats")

os.makedirs(CHATS_DIR, exist_ok=True)

MARKITDOWN = MarkItDown()

# Authenticated client for web_search / web_fetch (requires OLLAMA_API_KEY)
_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
if _OLLAMA_API_KEY:
    _client = Client(headers={"Authorization": f"Bearer {_OLLAMA_API_KEY}"})
else:
    _client = Client()


@dataclass
class ChatConfig:
    num_ctx: int = 2048
    temperature: float = 0.0 # usually 0.8
    top_p: float = 0.9 # usually 0.9
    top_k: int = 1 # usually around 40
    num_predict: int = 65536
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    seed: int = 0
    stop: str = ""
    min_p: float = 0.0
    think: bool = True
    enable_tools: bool = True

    def to_ollama_options(self) -> dict:
        return {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "seed": self.seed,
            "stop": [self.stop] if self.stop else [],
            "min_p": self.min_p,
        }


TOOL_MAP = {"web_search": _client.web_search, "web_fetch": _client.web_fetch}


def format_tool_results(results: Union[WebSearchResponse, WebFetchResponse, object], user_search: str) -> str:
    output = []
    if isinstance(results, WebSearchResponse):
        output.append(f'Search results for "{user_search}":')
        for result in results.results:
            output.append(f'{result.title}' if result.title else f'{result.content}')
            output.append(f'   URL: {result.url}')
            output.append(f'   Content: {result.content}')
            output.append('')
        return '\n'.join(output).rstrip()
    elif isinstance(results, WebFetchResponse):
        output.append(f'Fetch results for "{user_search}":')
        output.extend([
            f'Title: {results.title}',
            f'URL: {user_search}' if user_search else '',
            f'Content: {results.content}',
        ])
        if results.links:
            output.append(f'Links: {", ".join(results.links)}')
        output.append('')
        return '\n'.join(output).rstrip()
    else:
        return str(results)


def estimate_tokens(text: str) -> int:
    # 4 chars per token
    return len(text) // 4


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        # if msg.get("thinking"): # dont count thinking toks
        #     total += estimate_tokens(msg["thinking"])
        for tc in msg.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", {})
            total += estimate_tokens(json.dumps(args))
    return total


def convert_upload(uploaded) -> str:
    name = uploaded.name
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return MARKITDOWN.convert_stream(uploaded, file_extension=ext).text_content


def build_user_message(prompt: str, uploaded) -> tuple[dict, str | None]: #build query + context
    if uploaded is None:
        return {"role": "user", "content": prompt}, None

    name = uploaded.name
    try:
        converted = convert_upload(uploaded)
    except Exception as e:
        return None, f"Could not convert {name}: {e}"

    content = (
        f'Attached file: {name}\n'
        f'```markdown\n{converted}\n```\n\n'
        f'{prompt}'
    )
    message = {
        "role": "user",
        "content": content,
        "attachments": [{"name": name, "bytes": uploaded.size}],
    }
    return message, None


def _chat_path(chat_id: str) -> str:
    return os.path.join(CHATS_DIR, f"{chat_id}.json")


def _generate_chat_id() -> str:
    return uuid.uuid4().hex[:8]


def _chat_title(messages: list[dict]) -> str:
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"].strip().split("\n")[0][:40] or "New Chat"
    return "New Chat"


def list_saved_chats() -> list[dict]:
    chats = []
    for filename in os.listdir(CHATS_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(CHATS_DIR, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            chats.append({
                "id": data.get("id", filename[:-5]),
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", ""),
            })
        except Exception:
            continue
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


def load_chat(chat_id: str) -> list[dict]:
    path = _chat_path(chat_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("messages", [])


def save_chat(chat_id: str, messages: list[dict]) -> None:
    path = _chat_path(chat_id)
    data = {
        "id": chat_id,
        "title": _chat_title(messages),
        "messages": messages,
        "updated_at": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def delete_chat(chat_id: str) -> None:
    path = _chat_path(chat_id)
    if os.path.exists(path):
        os.remove(path)


def render_sidebar() -> ChatConfig:
    with st.sidebar:
        st.header("Chats")

        if st.button("+ New Chat", use_container_width=True):
            st.session_state.chat_id = _generate_chat_id()
            st.session_state.messages = []
            st.rerun()

        st.divider()

        saved = list_saved_chats()
        for chat in saved:
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                label = f"{chat['title']}"
                if st.button(label, key=f"load_{chat['id']}", use_container_width=True):
                    st.session_state.chat_id = chat["id"]
                    st.session_state.messages = load_chat(chat["id"])
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{chat['id']}", help="Delete chat"):
                    delete_chat(chat["id"])
                    if st.session_state.get("chat_id") == chat["id"]:
                        st.session_state.chat_id = _generate_chat_id()
                        st.session_state.messages = []
                    st.rerun()

        st.divider()
        st.header("Ollama Options")
        st.markdown(f"**Model:** {MODEL}")

        st.subheader("Load Time Options")
        num_ctx = st.number_input("num_ctx", min_value=1, value=ChatConfig.num_ctx, step=256)

        st.subheader("Context Usage")
        outgoing_preview = [{"role": "system", "content": SYSTEM_PROMPT}] + list(st.session_state.get("messages", []))
        used_tokens = estimate_messages_tokens(outgoing_preview)
        ctx_tokens = int(num_ctx)
        pct = min(used_tokens / ctx_tokens, 1.0) if ctx_tokens else 0.0
        st.progress(pct)
        st.caption(f"~{used_tokens:,} / {ctx_tokens:,} tokens ({pct:.0%})")

        st.subheader("Runtime Options")
        think = st.toggle("Enable thinking mode", value=ChatConfig.think)
        temperature = st.slider("temperature", min_value=0.0, max_value=2.0, value=ChatConfig.temperature, step=0.1)
        top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=ChatConfig.top_p, step=0.05)
        top_k = st.number_input("top_k", min_value=1, value=ChatConfig.top_k, step=1)
        num_predict = st.number_input("num_predict", value=ChatConfig.num_predict, step=1)
        repeat_penalty = st.number_input("repeat_penalty", min_value=0.0, value=ChatConfig.repeat_penalty, step=0.1, format="%.1f")
        repeat_last_n = st.number_input("repeat_last_n", value=ChatConfig.repeat_last_n, step=1)
        seed = st.number_input("seed", value=ChatConfig.seed, step=1)
        stop_str = st.text_input("stop", value=ChatConfig.stop)
        min_p = st.slider("min_p", min_value=0.0, max_value=1.0, value=ChatConfig.min_p, step=0.05)

        st.subheader("Agent Options")
        enable_tools = st.toggle("Enable web tools", value=ChatConfig.enable_tools)
        if enable_tools and not _OLLAMA_API_KEY:
            st.warning("OLLAMA_API_KEY not set — web tools will fail.")

        config = ChatConfig(
            num_ctx=int(num_ctx),
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            num_predict=int(num_predict),
            repeat_penalty=repeat_penalty,
            repeat_last_n=int(repeat_last_n),
            seed=int(seed),
            stop=stop_str,
            min_p=min_p,
            think=think,
            enable_tools=enable_tools,
        )

        st.json(config.to_ollama_options())

    return config


def render_messages(messages: list[dict]) -> None:
    for message in messages:
        role = message["role"]
        if role == "tool":
            continue
        with st.chat_message(role):
            if message.get("attachments"):
                st.caption("📎 " + ", ".join(a["name"] for a in message["attachments"]))
            if message.get("thinking"):
                with st.expander("Thinking", expanded=False):
                    st.code(message["thinking"])
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    name = tc.get("function", {}).get("name", "unknown")
                    args = tc.get("function", {}).get("arguments", {})
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    st.caption(f"🔧 Tool call: `{name}({args_str})`")
            if message.get("content"):
                st.markdown(message["content"])
            elif not message.get("thinking") and not message.get("tool_calls"):
                st.markdown("*No content*")


def stream_with_thinking(stream) -> tuple[str, str]:
    thinking_text = ""
    content_text = ""
    thinking_expander = st.expander("Thinking", expanded=False)
    with thinking_expander:
        thinking_placeholder = st.empty()
    content_placeholder = st.empty()

    for chunk in stream:
        msg = chunk["message"]
        if msg.get("thinking"):
            thinking_text += msg["thinking"]
            thinking_placeholder.code(thinking_text)
        if msg.get("content"):
            content_text += msg["content"]
            content_placeholder.markdown(content_text)

    return content_text, thinking_text


def stream_without_thinking(stream) -> str:
    return st.write_stream(chunk["message"]["content"] for chunk in stream)


def run_agent_turn(api_messages: list[dict], config: ChatConfig) -> tuple[str, str, list[dict]]:
    thinking_text = ""
    content_text = ""
    tool_calls = []

    thinking_placeholder = None
    if config.think:
        expander = st.expander("Thinking", expanded=False)
        with expander:
            thinking_placeholder = st.empty()

    content_placeholder = st.empty()

    stream = _client.chat(
        model=MODEL,
        messages=api_messages,
        stream=True,
        options=config.to_ollama_options(),
        think=config.think,
        tools=[_client.web_search, _client.web_fetch],
    )

    for chunk in stream:
        msg = chunk["message"]
        if msg.get("thinking"):
            thinking_text += msg["thinking"]
            if thinking_placeholder:
                thinking_placeholder.code(thinking_text)
        if msg.get("content"):
            content_text += msg["content"]
            content_placeholder.markdown(content_text)
        if msg.get("tool_calls"):
            tcs = msg["tool_calls"]
            if not isinstance(tcs, list):
                tcs = [tcs]
            tool_calls.extend(tcs)

    serializable_tool_calls = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            serializable_tool_calls.append(tc)
        else:
            serializable_tool_calls.append({
                "function": {
                    "name": tc.function.name,
                    "arguments": dict(tc.function.arguments),
                }
            })

    return content_text, thinking_text, serializable_tool_calls


def main() -> None:
    st.title("Chat with Ollama")
    config = render_sidebar()

    if "chat_id" not in st.session_state:
        st.session_state.chat_id = _generate_chat_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_messages(st.session_state.messages)

    uploaded = st.file_uploader("Attach a file", key=f"up_{len(st.session_state.messages)}")
    if prompt := st.chat_input("What do you want to ask?"):
        user_msg, error = build_user_message(prompt, uploaded)
        if error:
            st.error(error)
            st.stop()
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            if user_msg.get("attachments"):
                st.caption("📎 " + ", ".join(a["name"] for a in user_msg["attachments"]))
            st.markdown(user_msg["content"])

        with st.chat_message("assistant"):
            try:
                # inject system info and system prompt; and build conversation
                outgoing = list(st.session_state.messages)
                for i in range(len(outgoing) - 1, -1, -1):
                    if outgoing[i]["role"] == "user":
                        now = datetime.now().astimezone()
                        stamp = now.strftime(TIMESTAMP_FORMAT).strip()
                        outgoing[i] = {
                            **outgoing[i],
                            "content": f"{stamp}\n{outgoing[i]['content']}",
                        }
                        break
                if SYSTEM_PROMPT:
                    outgoing = [{"role": "system", "content": SYSTEM_PROMPT}] + outgoing

                if config.enable_tools:
                    # TOOL LOOP
                    while True:
                        content_text, thinking_text, tool_calls = run_agent_turn(outgoing, config)
                        assistant_msg = {
                            "role": "assistant",
                            "content": content_text,
                            "thinking": thinking_text,
                        }
                        if tool_calls:
                            assistant_msg["tool_calls"] = tool_calls

                        st.session_state.messages.append(assistant_msg)
                        outgoing.append(assistant_msg)

                        if not tool_calls:
                            break

                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            args = tc["function"]["arguments"]
                            tool_fn = TOOL_MAP.get(name)
                            if tool_fn:
                                st.caption(f"🔧 Running {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})...")
                                try:
                                    result = tool_fn(**args)
                                    formatted = format_tool_results(result, args.get("query", "") or args.get("url", ""))
                                    capped = formatted[:TOOL_RESULT_LIMIT]
                                except Exception as e:
                                    capped = f"Error calling {name}: {e}"
                            else:
                                capped = f"Tool {name} not found."

                            tool_msg = {
                                "role": "tool",
                                "content": capped,
                                "tool_name": name,
                            }
                            st.session_state.messages.append(tool_msg)
                            outgoing.append(tool_msg)
                            # st.caption(f"✅ {name} done")
                else:
                    stream = _client.chat(
                        model=MODEL,
                        messages=outgoing,
                        stream=True,
                        options=config.to_ollama_options(),
                        think=config.think,
                    )
                    if config.think:
                        content_text, thinking_text = stream_with_thinking(stream)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": content_text, "thinking": thinking_text}
                        )
                    else:
                        response = stream_without_thinking(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                save_chat(st.session_state.chat_id, st.session_state.messages)
            except Exception as e:
                st.error(f"Ollama error: {e}")


if __name__ == "__main__":
    main()
