from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from prompts import ARCHITECT_SYSTEM, CODER_SYSTEM, REVIEWER_SYSTEM
from schemas import ArchitectResult, CoderResult, ReviewResult
from tools import list_files, read_file, write_file, run_tests

MAX_ITERATIONS = 10
DEFAULT_MODEL = "qwen3.5:35b-a3b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"

Provider = Literal["ollama", "gigachat"]




class AgentState(TypedDict, total=False):
    user_task: str
    repo_path: str
    repo_context: str

    architect_plan: str
    target_files: list[str]
    files_to_read_first: list[str]
    acceptance_criteria: list[str]

    coder_summary: str
    changed_files: list[str]
    test_output: str

    review_status: Literal["approved", "changes_requested"] | None
    review_notes: str

    iteration: int
    final_answer: str


def build_llm(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    provider: Provider = "ollama",
    think: bool = False,
) -> BaseChatModel:
    if provider == "gigachat":
        from langchain_gigachat import GigaChat
        return GigaChat(
            model=model,
            verify_ssl_certs=False,
            profanity_check=False,
            timeout=300,
        )
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key="ollama",
        temperature=0,
        extra_body={"think": think},
    )


def build_graph(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    provider: Provider = "ollama",
) -> object:
    llm = build_llm(model=model, base_url=base_url, provider=provider)

    def _invoke_structured(schema, prompt: str, node: str):
        """Invoke with_structured_output and retry up to 3 times if None is returned."""
        structured_llm = llm.with_structured_output(schema, method="function_calling")
        for attempt in range(3):
            result = structured_llm.invoke(prompt)
            if result is not None:
                return result
            print(f"[{node}] structured output returned None, retrying ({attempt + 1}/3)...")
        raise RuntimeError(f"[{node}] LLM returned None after 3 attempts — model did not call the function.")

    def architect_node(state: AgentState) -> dict:
        result: ArchitectResult = _invoke_structured(
            ArchitectResult,
            f"{ARCHITECT_SYSTEM}\n\n"
            f"Задача пользователя:\n{state['user_task']}\n\n"
            f"Контекст проекта:\n{state.get('repo_context', '')}\n\n"
            f"Дерево файлов:\n{list_files(state.get('repo_path', ''))}",
            "architect",
        )
        plan_text = "\n".join([
            f"Summary: {result.summary}",
            "Files to read first:",
            *[f"- {x}" for x in result.files_to_read_first],
            "Target files:",
            *[f"- {x}" for x in result.target_files],
            "Acceptance criteria:",
            *[f"- {x}" for x in result.acceptance_criteria],
            "Implementation notes:",
            *[f"- {x}" for x in result.implementation_notes],
        ])
        return {
            "architect_plan": plan_text,
            "target_files": result.target_files,
            "files_to_read_first": result.files_to_read_first,
            "acceptance_criteria": result.acceptance_criteria,
            "iteration": 0,
        }

    def coder_node(state: AgentState) -> dict:
        repo_path = state.get("repo_path", "")
        files_to_read = state.get("files_to_read_first") or state.get("target_files") or []

        file_contents = [
            f"\n===== FILE: {path} =====\n{read_file(repo_path, path)}\n"
            for path in files_to_read[:8]
        ]

        prompt = (
            f"{CODER_SYSTEM}\n\n"
            f"Задача:\n{state['user_task']}\n\n"
            f"План архитектора:\n{state.get('architect_plan', '')}\n\n"
            f"Критерии приемки:\n{state.get('acceptance_criteria', [])}\n\n"
            f"Замечания ревьюера:\n{state.get('review_notes', '')}\n\n"
            f"Ниже содержимое релевантных файлов:\n{''.join(file_contents)}"
        )

        result: CoderResult = _invoke_structured(CoderResult, prompt, "coder")

        changed_files = result.changed_files
        write_logs = [
            write_file(repo_path, w.path, w.content)
            for w in result.writes
        ]

        test_output = run_tests(repo_path) if result.run_tests else ""

        full_summary = "\n".join([
            result.summary, "",
            "Changed files:", *[f"- {x}" for x in changed_files], "",
            "Write results:", *[f"- {x}" for x in write_logs], "",
            "Test output:", test_output or "(tests not run)",
        ])

        return {
            "coder_summary": full_summary,
            "changed_files": changed_files,
            "test_output": test_output,
        }

    def reviewer_node(state: AgentState) -> dict:
        repo_path = state.get("repo_path", "")
        changed_files = state.get("changed_files") or []

        # Limit per-file content to avoid token overflow
        FILE_CHARS = 3000
        TOTAL_CHARS = 12000
        raw_contents = []
        total = 0
        for path in changed_files[:8]:
            content = read_file(repo_path, path)[:FILE_CHARS]
            chunk = f"\n===== FILE: {path} =====\n{content}\n"
            if total + len(chunk) > TOTAL_CHARS:
                break
            raw_contents.append(chunk)
            total += len(chunk)

        result: ReviewResult = _invoke_structured(
            ReviewResult,
            f"{REVIEWER_SYSTEM}\n\n"
            f"Задача:\n{state['user_task']}\n\n"
            f"План архитектора:\n{state.get('architect_plan', '')}\n\n"
            f"Критерии приемки:\n{state.get('acceptance_criteria', [])}\n\n"
            f"Измененные файлы:\n{changed_files}\n\n"
            f"Содержимое измененных файлов:\n{''.join(raw_contents)}\n\n"
            f"Отчет кодера:\n{state.get('coder_summary', '')}\n\n"
            f"Результат тестов:\n{state.get('test_output', '')}",
            "reviewer",
        )

        return {
            "review_status": result.status,
            "review_notes": "\n".join(f"- {x}" for x in result.issues),
        }

    def rewrite_gate_node(state: AgentState) -> dict:
        return {"iteration": state.get("iteration", 0) + 1}

    def finalize_node(state: AgentState) -> dict:
        return {"final_answer": "\n".join([
            "# Architect plan",
            state.get("architect_plan", ""), "",
            "# Coder summary",
            state.get("coder_summary", ""), "",
            "# Reviewer verdict",
            f"status: {state.get('review_status', 'unknown')}",
            state.get("review_notes", ""),
        ])}

    def route_after_review(state: AgentState) -> str:
        if state.get("review_status") == "approved":
            return "finalize"
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "finalize"
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("architect", architect_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("rewrite", rewrite_gate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("architect")
    graph.add_edge("architect", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_edge("rewrite", "coder")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"rewrite": "rewrite", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()
