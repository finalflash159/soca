DISPLAY_NAME = "SoCa"
COMPACT_MARK = DISPLAY_NAME

# Compact bird mark shown at the left of the splash; info lines sit to its right
# (Claude Code style), instead of a large figlet wordmark.
BIRD_MARK = "\n".join(
    [
        "   .-.",
        " .(//)>",
        " ///_/",
        "  //'",
    ]
)

# Voice-mode poses. The bird is calm/perched while idle or listening, and
# "sings" (open beak + floating notes) while SoCa speaks. Same line count so the
# response cell does not jump when the state changes.
BIRD_IDLE = "\n".join(
    [
        "    .-.",
        "  .(//)>",
        "  ///_/",
        "   //'",
    ]
)

BIRD_SINGING = "\n".join(
    [
        " ♪  .-.   ♫",
        "  .(o/)>  ♪",
        "  ///_/",
        "   //'",
    ]
)


def compact_header(
    *,
    mode: str,
    profile: str,
    llm_model: str | None = None,
    vault_status: str = "unknown",
    memory_status: str = "unknown",
    runtime_state: str = "idle",
) -> str:
    parts = [
        COMPACT_MARK,
        mode,
        f"profile={profile}",
    ]
    if llm_model:
        parts.append(f"LLM={llm_model}")
    parts.extend(
        [
            f"vault={vault_status}",
            f"memory={memory_status}",
            runtime_state,
        ]
    )
    return "  ".join(parts)


__all__ = [
    "BIRD_IDLE",
    "BIRD_MARK",
    "BIRD_SINGING",
    "COMPACT_MARK",
    "DISPLAY_NAME",
    "compact_header",
]
