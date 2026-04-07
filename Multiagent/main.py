import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from graph import build_graph

PROVIDER = os.environ.get("PROVIDER", "ollama")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:35b-a3b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

GIGACHAT_MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat-2-Pro")

TASK_FILE = Path(__file__).parent / "task.md"


def load_task(path: Path) -> dict:
    """Parse task.md into sections by '# heading' markers."""
    sections: dict[str, str] = {}
    current = None
    lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = line[2:].strip()
            lines = []
        else:
            lines.append(line)

    if current is not None:
        sections[current] = "\n".join(lines).strip()

    return sections


def main():
    if not TASK_FILE.exists():
        print(f"Ошибка: файл задачи не найден: {TASK_FILE}")
        sys.exit(1)

    task = load_task(TASK_FILE)

    user_task = task.get("user_task", "")
    repo_context = task.get("repo_context", "")
    repo_path = task.get("repo_path", str(Path(__file__).parent))

    if not user_task:
        print("Ошибка: секция '# user_task' пустая или отсутствует в task.md")
        sys.exit(1)

    if PROVIDER == "gigachat":
        creds = os.environ.get("GIGACHAT_CREDENTIALS", "")
        if not creds:
            print("Ошибка: задайте GIGACHAT_CREDENTIALS в файле .env")
            sys.exit(1)
        # Auto-fix: if credentials are in "client_id:base64" format, keep only the base64 part
        import base64
        if ":" in creds:
            parts = creds.split(":", 1)
            try:
                base64.b64decode(parts[1], validate=True)
                print("Внимание: GIGACHAT_CREDENTIALS содержит 'client_id:ключ' — используется только base64-часть.")
                os.environ["GIGACHAT_CREDENTIALS"] = parts[1]
            except Exception:
                pass  # Not that format, leave as-is
        app = build_graph(model=GIGACHAT_MODEL, provider="gigachat")
    else:
        app = build_graph(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, provider="ollama")

    result = app.invoke(
        {
            "user_task": user_task,
            "repo_context": repo_context,
            "repo_path": repo_path,
        }
    )

    print("\n" + "=" * 80)
    print(result["final_answer"])
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
