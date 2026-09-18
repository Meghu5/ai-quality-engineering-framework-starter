import pytest


@pytest.mark.agents
def test_agent_uses_allowed_tool(agent_client):
    result = agent_client.run("Find my order status for order 123")
    assert result.tool_name == "get_order_status"
    assert result.arguments["order_id"] == "123"
    assert result.completed is True
