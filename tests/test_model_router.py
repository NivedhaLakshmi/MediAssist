"""Model-driven source selection falls back safely.

The router asks the language model which medical source fits a
question. These tests cover what happens when it does not get a
usable answer, since every one of those paths must end in the
keyword rules rather than in a failed question.

The module is loaded from its path rather than by adding the MCP
server to the import path. That directory contains its own ``utils``
package, which would shadow the application's and break unrelated
tests.
"""

import ast
import importlib.util
import os
import sys
import types

import pytest


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ROUTER_PATH = os.path.join(
    PROJECT_ROOT, "medical_mcp", "utils", "model_router.py"
)

SEARCH_PATH = os.path.join(
    PROJECT_ROOT, "medical_mcp", "tools", "search.py"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_router = _load(ROUTER_PATH, "mcp_model_router")


def _fake_ollama(reply=None, error=None):
    """A stand-in for the ollama package returning one fixed reply."""

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *args, **kwargs):
            if error:
                raise error
            return reply

    module = types.ModuleType("ollama")
    module.Client = FakeClient

    return module


@pytest.fixture
def routing_on(monkeypatch):
    monkeypatch.setattr(model_router, "ROUTING_MODE", "model")


def _install(monkeypatch, **kwargs):
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama(**kwargs))


def _tool_reply(*names):
    return {
        "message": {
            "tool_calls": [
                {"function": {"name": name}} for name in names
            ]
        }
    }


def test_returns_the_tool_the_model_chose(monkeypatch, routing_on):
    _install(monkeypatch, reply=_tool_reply("search_dailymed"))

    assert model_router.choose_source(
        "what are the side effects of metformin"
    ) == "search_dailymed"


def test_falls_back_when_routing_is_switched_off(monkeypatch):
    monkeypatch.setattr(model_router, "ROUTING_MODE", "rules")
    _install(monkeypatch, reply=_tool_reply("search_dailymed"))

    assert model_router.choose_source("anything") is None


def test_falls_back_when_the_runtime_is_unreachable(
    monkeypatch, routing_on
):
    _install(monkeypatch, error=ConnectionError("no runtime"))

    assert model_router.choose_source("what is anaemia") is None


def test_falls_back_when_the_model_names_no_tool(
    monkeypatch, routing_on
):
    _install(
        monkeypatch,
        reply={"message": {"content": "I am not sure."}},
    )

    assert model_router.choose_source("what is anaemia") is None


def test_falls_back_when_the_model_names_several_tools(
    monkeypatch, routing_on
):
    _install(
        monkeypatch,
        reply=_tool_reply("search_dailymed", "search_pubmed"),
    )

    assert model_router.choose_source("what is anaemia") is None


def test_falls_back_when_the_model_invents_a_tool(
    monkeypatch, routing_on
):
    _install(monkeypatch, reply=_tool_reply("search_webmd"))

    assert model_router.choose_source("what is anaemia") is None


def test_falls_back_when_the_tool_call_is_malformed(
    monkeypatch, routing_on
):
    _install(
        monkeypatch,
        reply={"message": {"tool_calls": [{"nope": {}}]}},
    )

    assert model_router.choose_source("what is anaemia") is None


@pytest.mark.parametrize("question", ["", "   ", None])
def test_an_empty_question_never_reaches_the_model(
    monkeypatch, routing_on, question
):
    _install(monkeypatch, error=AssertionError("must not be called"))

    assert model_router.choose_source(question) is None


def _searcher_names_in_source():
    """SEARCHERS keys, read statically so nothing has to import."""

    tree = ast.parse(open(SEARCH_PATH).read())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            t.id for t in node.targets if isinstance(t, ast.Name)
        ]
        if "SEARCHERS" in targets and isinstance(node.value, ast.Dict):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
            }

    raise AssertionError("SEARCHERS not found in search.py")


def test_every_advertised_tool_is_a_real_searcher():
    """A schema the model can pick must map to something callable."""

    advertised = {
        schema["function"]["name"]
        for schema in model_router.TOOL_SCHEMAS
    }

    assert advertised == set(model_router.KNOWN_TOOLS)
    assert advertised == _searcher_names_in_source()
