"""配置：全部从环境变量读，API Key 只放在 .env（不入库）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v[:1] not in ('"', "'"):
            v = v.split(" #", 1)[0].strip()  # 未加引号的值去掉行尾注释（.env.example 里就有）
        os.environ.setdefault(k.strip(), v.strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_mode: str            # offline | deepseek
    deepseek_api_key: str
    deepseek_base_url: str
    model_pro: str             # 主智能体、故障诊断
    model_flash: str           # 告警分诊
    mcp_transport: str         # stdio | http
    mcp_url: str
    wecom_webhook_url: str
    offline_step_delay: float  # 离线模式下每步停顿，方便看清推理过程
    skills_dir: Path


def get_settings() -> Settings:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    mode = os.environ.get("MODEL_MODE", "auto").lower()
    if mode == "auto":
        mode = "deepseek" if key else "offline"
    return Settings(
        model_mode=mode,
        deepseek_api_key=key,
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model_pro=os.environ.get("DEEPSEEK_MODEL_PRO", "deepseek-v4-pro"),
        model_flash=os.environ.get("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash"),
        mcp_transport=os.environ.get("MCP_TRANSPORT", "stdio"),
        mcp_url=os.environ.get("MCP_URL", "http://127.0.0.1:8765/mcp"),
        wecom_webhook_url=os.environ.get("WECOM_WEBHOOK_URL", ""),
        offline_step_delay=float(os.environ.get("OFFLINE_STEP_DELAY", "0")),
        skills_dir=ROOT / "skills",
    )
