from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel, Field

# 泛型类型，支持任意数据类型
T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):
    """统一响应模型"""
    code: int = Field(200, description="状态码，200为成功，非200为失败")
    message: str = Field("success", description="响应信息，成功为success，失败为错误原因")
    data: Optional[T] = Field(None, description="响应数据体")

    @classmethod
    def success(cls, data: T = None, message: str = "success") -> "ResponseModel[T]":
        """成功响应"""
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int = 400, message: str = "请求失败", data: T = None) -> "ResponseModel[T]":
        """失败响应"""
        return cls(code=code, message=message, data=data)

# 通用分页请求模型
class PageQuery(BaseModel):
    page: int = Field(1, ge=1, description="页码，从1开始")
    page_size: int = Field(10, ge=1, le=100, description="每页条数，1-100")

# 通用分页响应模型
class PageResult(BaseModel, Generic[T]):
    total: int = Field(description="总条数")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页条数")
    total_page: int = Field(description="总页数")
    list: List[T] = Field(description="数据列表")
