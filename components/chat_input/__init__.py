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


def chat_input(value="", voice_completion_id=0):
    # The key must be stable: without one Streamlit derives the widget
    # id from the element's position, which the growing chat history
    # shifts, remounting the component and discarding its pending value.
    return _chat_input(
        value=value,
        voice_completion_id=voice_completion_id,
        default=None,
        key="mediassist_chat_input",
    )
