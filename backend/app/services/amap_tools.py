from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from app.config import get_settings

from ..logging_config import get_logger

logger = get_logger(__name__)

_client: MultiServerMCPClient | None = None
_tools: list[BaseTool] = []


async def init_amap_tools() -> None:
    global _client, _tools
    s = get_settings()
    if not s.amap_api_key:
        logger.warning("AMAP_API_KEY 未配置，Amap 工具不可用")
        return
    _client = MultiServerMCPClient({
        "amap": {
            "command": "uvx",
            "args": ["amap-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": s.amap_api_key},
            "transport": "stdio",
        }
    })
    _tools = await _client.get_tools()
    logger.info("Amap MCP 工具初始化成功，共 %d 个工具", len(_tools))


async def close_amap_tools() -> None:
    global _client
    _client = None


def get_amap_tools() -> list[BaseTool]:
    return _tools


def get_tool_by_name(name: str) -> BaseTool | None:
    return next((t for t in _tools if name in t.name), None)
