"""FastAPI主应用"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ..config import get_settings, validate_config, log_config
from ..logging_config import get_logger, new_request_id, set_request_id, setup_logging
from ..services.amap_tools import init_amap_tools, close_amap_tools
from .routes import trip, poi, chat as chat_routes

# 获取配置
settings = get_settings()

setup_logging()
logger = get_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于HelloAgents框架的智能旅行规划助手API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """给每个请求分配 ID，贯穿该请求的所有日志，并回写响应头便于前后端对账。"""
    request_id = request.headers.get("X-Request-ID") or new_request_id()
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# 注册路由
app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")
app.include_router(chat_routes.router, prefix="/api")


def _setup_langsmith():
    import os
    s = get_settings()
    if s.langchain_tracing_v2 == "true" and s.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = s.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = s.langchain_project
        logger.info("LangSmith 追踪已启用，项目：%s", s.langchain_project)
    else:
        logger.info("LangSmith 追踪未启用")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("启动 %s v%s", settings.app_name, settings.app_version)
    _setup_langsmith()
    log_config()
    try:
        validate_config()
        logger.info("配置验证通过")
    except ValueError as e:
        logger.error("配置验证失败: %s", e)
        raise
    await init_amap_tools()
    from ..services.checkpointer import init_checkpointer
    from ..agents import init_trip_graph
    checkpointer = await init_checkpointer()
    init_trip_graph(checkpointer)
    logger.info("启动完成，API 文档: http://%s:%s/docs", settings.host, settings.port)


@app.on_event("shutdown")
async def shutdown_event():
    await close_amap_tools()
    from ..services.checkpointer import close_checkpointer
    await close_checkpointer()
    logger.info("应用已关闭")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

