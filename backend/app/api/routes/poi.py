"""POI相关API路由"""

from fastapi import APIRouter, HTTPException

from ...services.unsplash_service import get_unsplash_service

from ...logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/poi", tags=["POI"])


@router.get(
    "/photo",
    summary="获取景点图片",
    description="根据景点名称从Unsplash获取图片"
)
async def get_attraction_photo(name: str):
    """
    获取景点图片

    Args:
        name: 景点名称

    Returns:
        图片URL
    """
    try:
        unsplash_service = get_unsplash_service()

        # 搜索景点图片
        photo_url = unsplash_service.get_photo_url(f"{name} China landmark")

        if not photo_url:
            # 如果没找到,尝试只用景点名称搜索
            photo_url = unsplash_service.get_photo_url(name)

        return {
            "success": True,
            "message": "获取图片成功",
            "data": {
                "name": name,
                "photo_url": photo_url
            }
        }

    except Exception as e:
        logger.error("获取景点图片失败: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"获取景点图片失败: {str(e)}"
        )
