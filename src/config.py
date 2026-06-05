"""
全局配置管理 — 从 .env 文件加载所有配置。

怎么用：
    from src.config import settings
    print(settings.llm.model)       # → "gpt-4o-mini"
    print(settings.storage.data_dir)  # → Path("./data")

为什么用 dataclass？
    比字典强在有代码提示，IDE 能自动补全 settings.llm.xxx
    写错了会直接报错，不会悄悄返回 None
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 把 .env 文件里的 KEY=VALUE 加载为环境变量
load_dotenv()

# 项目根目录 = mindflow/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMConfig:
    """LLM 服务配置 — 控制使用哪个模型、API 地址"""
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )


@dataclass
class BotConfig:
    """Telegram Bot 配置"""
    token: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "")
    )


@dataclass
class StorageConfig:
    """存储配置 — 控制数据和数据库存在哪里"""
    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    )
    chroma_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data") / "chroma"
    )
    db_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data") / "mindflow.db"
    )

    def __post_init__(self):
        """初始化后自动创建必需的目录"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """全局配置入口 — 把上面三个配置合在一起"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


# 全局单例 — 整个项目只有这一个 Settings 实例
# 其他模块直接 from src.config import settings 即可
settings = Settings()
