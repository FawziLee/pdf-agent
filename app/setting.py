# from pydantic_settings import BaseSettings
# from pydantic import Field
# import os
# from pathlib import Path

# # 项目根目录
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR = PROJECT_ROOT / "data"
# UPLOAD_DIR = DATA_DIR / "upload"


# class Settings(BaseSettings):
#     qwen_api_key: str = Field(default="", env="QWEN_API_KEY")
#     paddle_token: str = Field(default="", env="PADDLE_TOKEN")

#     # 上传限制
#     allowed_file_extensions: list[str] = Field(
#         default_factory=lambda: ["pdf", "png", "jpg", "jpeg"]
#     )
#     max_upload_file_size: int = Field(default=200 * 1024 * 1024)  # 200MB



# settings = Settings()

# # 确保目录存在
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
