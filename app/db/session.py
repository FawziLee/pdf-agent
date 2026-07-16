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


def _ensure_sqlite_columns():
    """给已有 SQLite 表补齐新增列（create_all 不会 ALTER）。"""
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(documents)").fetchall()
        existing = {row[1] for row in rows}  # row[1] = column name
        if "summary" not in existing:
            conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN summary TEXT")
            logger.info("已为 documents 表添加 summary 列")
        if "section_summaries" not in existing:
            conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN section_summaries TEXT")
            logger.info("已为 documents 表添加 section_summaries 列")


def init_db():
    """项目启动时初始化数据库表"""
    try:
        from app.models.document import Document  # noqa: F401

        Base.metadata.create_all(bind=engine)
        _ensure_sqlite_columns()
        logger.info(f"SQLite 初始化成功：{SQLALCHEMY_DATABASE_URL}")
    except Exception as e:
        logger.error(f"数据库表初始化失败：{str(e)}", exc_info=True)
        raise
