from app.core.config import get_settings
from app.gateway.router import GatewayRouter
from app.llm.factory import get_provider
from app.llm.stubs import AnthropicProvider


def test_mock_gateway():
    settings = get_settings()
    settings.llm_provider = "mock"
    gw = GatewayRouter(settings)
    assert gw.mode == "mock"
    r = gw.complete(system="s", user="hello", mock_builder=lambda: "world")
    assert r.text == "world"
    est = gw.compare_estimate("hello world", 200)
    assert len(est) >= 3


def test_anthropic_requires_key():
    p = AnthropicProvider(None)
    try:
        p.complete(system="s", user="u")
        assert False, "should have raised"
    except RuntimeError as e:
        assert "ANTHROPIC_API_KEY" in str(e)


def test_factory_mock():
    settings = get_settings()
    settings.llm_provider = "mock"
    p = get_provider(settings)
    assert p.name == "mock"
