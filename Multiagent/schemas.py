from typing import Literal
from pydantic import BaseModel, Field


class ArchitectResult(BaseModel):
    """Результат работы архитектора: план реализации задачи."""

    summary: str = Field(description="Краткое описание решения")
    target_files: list[str] = Field(default_factory=list, description="Файлы, которые нужно изменить")
    files_to_read_first: list[str] = Field(default_factory=list, description="Файлы, которые нужно сначала прочитать")
    acceptance_criteria: list[str] = Field(default_factory=list, description="Критерии приемки")
    implementation_notes: list[str] = Field(default_factory=list, description="Технические замечания по реализации")


class ReviewResult(BaseModel):
    """Результат код-ревью: статус и список замечаний."""

    status: Literal["approved", "changes_requested"]
    issues: list[str] = Field(default_factory=list, description="Конкретные замечания по коду")

class FileWrite(BaseModel):
    """Файл для записи на диск."""

    path: str = Field(description="Относительный путь к файлу")
    content: str = Field(description="Полное новое содержимое файла")


class CoderResult(BaseModel):
    """Результат работы кодера: изменения и файлы для записи."""

    summary: str = Field(description="Краткое описание сделанных изменений")
    changed_files: list[str] = Field(default_factory=list, description="Список изменённых файлов (относительные пути)")
    writes: list[FileWrite] = Field(default_factory=list, description="Файлы для записи на диск")
    run_tests: bool = Field(default=False, description="True если нужно запустить тесты после записи")

class CodingState(BaseModel):
    user_task: str
    repo_path: str
    repo_context: str = ""

    architect_plan: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    files_to_read_first: list[str] = Field(default_factory=list)
    coder_summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    test_output: str = ""
  
    review_notes: str = ""
    review_status: Literal["approved", "changes_requested"] | None = None
    final_answer: str = ""
    iteration: int = 0