import os
from pathlib import Path

solution = Path(__file__).resolve().parents[2]
home = solution / "tmp" / "work" / "runtime" / "codex-home"
home.mkdir(parents=True, exist_ok=True, mode=0o700)
os.chmod(home, 0o700)


def write_private(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    os.chmod(path, 0o600)


write_private(home / "config.toml", (solution / "config" / "codex" / "config.toml").read_bytes())
for role, model in (("answer", "gpt-6-sol"), ("answer", "gpt-6-luna"), ("judge", "gpt-6-sol")):
    instructions = solution / "prompts" / "roles" / role / "base.md"
    profile = f"{role}-{model.removeprefix('gpt-6-')}.config.toml"
    content = (
        f'model = "{model}"\n'
        'model_reasoning_effort = "high"\n'
        f'model_instructions_file = "{instructions}"\n'
    )
    write_private(home / profile, content.encode())
