import os
import streamlit.components.v1 as components


_COMPONENT_PATH = os.path.join(
    os.path.dirname(__file__),
    "frontend"
)


_chat_input = components.declare_component(
    "mediassist_chat_input",
    path=_COMPONENT_PATH
)


def chat_input(value=""):
    return _chat_input(
        value=value,
        default=None
    )