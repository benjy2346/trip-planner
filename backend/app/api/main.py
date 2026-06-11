"""FastAPI主应用"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config import get_settings, validate_config, print_config
from ..services.amap_tools import init_amap_tools, close_amap_tools
from .routes import trip, poi, map as map_routes, chat as chat_routes

# 获取配置
settings = get_settings()

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

# 注册路由
app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")
app.include_router(map_routes.router, prefix="/api")
app.include_router(chat_routes.router, prefix="/api")


def _setup_langsmith():
    import os
    s = get_settings()
    if s.langchain_tracing_v2 == "true" and s.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = s.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = s.langchain_project
        print(f"✅ LangSmith 追踪已启用，项目：{s.langchain_project}")
    else:
        print("ℹ️  LangSmith 追踪未启用")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    print("\n" + "="*60)
    print(f"🚀 {settings.app_name} v{settings.app_version}")
    print("="*60)
    _setup_langsmith()
    print_config()
    try:
        validate_config()
        print("\n✅ 配置验证通过")
    except ValueError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
        raise
    await init_amap_tools()
    from ..services.checkpointer import init_checkpointer
    from ..agents import init_supervisor_graph
    checkpointer = await init_checkpointer()
    init_supervisor_graph(checkpointer)
    print("\n" + "="*60)
    print("📚 API文档: http://localhost:8000/docs")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    await close_amap_tools()
    from ..services.checkpointer import close_checkpointer
    await close_checkpointer()
    print("\n👋 应用已关闭")


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

