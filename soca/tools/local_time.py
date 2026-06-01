from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from soca.tools.base import SideEffectLevel, ToolResult, ToolSpec, object_schema


class LocalTimeTool:
    def __init__(
        self,
        default_timezone: str = "Asia/Ho_Chi_Minh",
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.default_timezone = default_timezone
        self.now_fn = now_fn or (lambda: datetime.now(tz=ZoneInfo(self.default_timezone)))

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="local_time.now",
            description="Return the current local date and time.",
            input_schema=object_schema(
                properties={
                    "timezone": {
                        "type": "string",
                        "description": "Optional IANA timezone name.",
                    }
                }
            ),
            side_effect=SideEffectLevel.READ_ONLY,
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        timezone_name = str(arguments.get("timezone") or self.default_timezone)
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone: {timezone_name}") from exc

        now = self.now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone)
        else:
            now = now.astimezone(timezone)

        content = f"Bây giờ là {now:%H:%M}, ngày {now:%d/%m/%Y}."
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=content,
            data={
                "iso": now.isoformat(),
                "timezone": timezone_name,
                "hour": now.hour,
                "minute": now.minute,
                "date": now.date().isoformat(),
            },
        )
