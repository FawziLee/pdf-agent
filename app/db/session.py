from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

# 项目根目录 / data / pdf_agent.db
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "pdf_agent.db"

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# SQLite 需要 check_same_thread=False 才能在 FastAPI 中使用
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类，所有 ORM 模型都继承这个类
Base = declarative_base()


def get_db() -> Session:
    """FastAPI 依赖：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """项目启动时初始化数据库表"""
    try:
        from app.models.document import Document  # noqa: F401

        Base.metadata.create_all(bind=engine)
        logger.info(f"SQLite 初始化成功：{SQLALCHEMY_DATABASE_URL}")
    except Exception as e:
        logger.error(f"数据库表初始化失败：{str(e)}", exc_info=True)
        raise
