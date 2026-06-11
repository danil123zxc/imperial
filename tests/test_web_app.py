import json
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace

from langchain_core.documents import Document

from imperial_rag import web_app
from imperial_rag.web_app import (
    APP_TITLE,
    RetrievedFileGroup,
    build_retrieved_file_groups,
    build_status_summary,
    load_status_summary,
)


def test_status_summary_displays_manifest_counts():
    summary = build_status_summary(total_files=162, indexed_files=100, failed_files=3)

    assert APP_TITLE == "Imperial RAG"
    assert "Total files: 162" in summary
    assert "Indexed files: 100" in summary
    assert "Failed files: 3" in summary


def test_load_status_summary_is_importable_without_manifest_stack():
    summary = load_status_summary(settings=object())

    assert "Total files:" in summary


def test_build_retrieved_file_groups_groups_chunks_by_file_and_loads_file_preview(tmp_path):
    documents_root = tmp_path / "documents"
    extraction_root = tmp_path / ".imperial_rag" / "extracted"
    source_path = documents_root / "11. РЕГЛАМЕНТЫ" / "Регламент ЛОГИСТИКА.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"docx bytes")
    artifact_path = extraction_root / "documents" / "logistics.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"page_content": "Первый фрагмент полного файла."},
                    {"page_content": "Второй фрагмент полного файла."},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=extraction_root)

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Retrieved chunk should not be shown.",
                metadata={
                    "file_id": "logistics",
                    "file_path": str(source_path),
                    "relative_path": "11. РЕГЛАМЕНТЫ/Регламент ЛОГИСТИКА.docx",
                    "file_name": "Регламент ЛОГИСТИКА.docx",
                    "source_type": "body",
                },
            ),
            Document(
                page_content="Another retrieved chunk should not be shown.",
                metadata={
                    "file_id": "logistics",
                    "file_path": str(source_path),
                    "relative_path": "11. РЕГЛАМЕНТЫ/Регламент ЛОГИСТИКА.docx",
                    "file_name": "Регламент ЛОГИСТИКА.docx",
                    "source_type": "body",
                },
            ),
        ],
        settings,
    )

    assert len(groups) == 1
    assert groups[0].file_name == "Регламент ЛОГИСТИКА.docx"
    assert groups[0].display_path == "11. РЕГЛАМЕНТЫ/Регламент ЛОГИСТИКА.docx"
    assert groups[0].preview_text == "Первый фрагмент полного файла.\n\nВторой фрагмент полного файла."
    assert not hasattr(groups[0], "snippets")
    assert not hasattr(groups[0], "markers")
    assert not hasattr(groups[0], "chunk_count")
    assert groups[0].can_download is True


def test_build_retrieved_file_groups_groups_same_relative_path_without_matching_file_id(tmp_path):
    documents_root = tmp_path / "documents"
    source_path = documents_root / "docs" / "same.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"same")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="First chunk",
                metadata={
                    "file_id": "old-id",
                    "relative_path": "docs/same.docx",
                    "file_name": "same.docx",
                },
            ),
            Document(
                page_content="Second chunk",
                metadata={
                    "relative_path": "docs/same.docx",
                    "file_name": "same.docx",
                },
            ),
        ],
        settings,
    )

    assert len(groups) == 1
    assert groups[0].file_key == "relative_path:docs/same.docx"


def test_build_retrieved_file_groups_keeps_same_filename_in_different_folders_separate(tmp_path):
    documents_root = tmp_path / "documents"
    first = documents_root / "sales" / "policy.docx"
    second = documents_root / "hr" / "policy.docx"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"sales")
    second.write_bytes(b"hr")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Sales policy",
                metadata={"relative_path": "sales/policy.docx", "file_name": "policy.docx"},
            ),
            Document(
                page_content="HR policy",
                metadata={"relative_path": "hr/policy.docx", "file_name": "policy.docx"},
            ),
        ],
        settings,
    )

    assert [group.display_path for group in groups] == ["sales/policy.docx", "hr/policy.docx"]


def test_normalize_retrieved_file_groups_merges_stored_duplicate_cards(tmp_path):
    documents_root = tmp_path / "documents"
    extraction_root = tmp_path / "extracted"
    source_path = documents_root / "docs" / "same.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"same")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=extraction_root)
    first = RetrievedFileGroup(
        file_key="file_id:old",
        file_name="same.docx",
        display_path="docs/same.docx",
        download_path=source_path,
        download_name="same.docx",
        download_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        preview_text="First preview",
        can_download=True,
    )
    second = RetrievedFileGroup(
        file_key="relative_path:docs/same.docx",
        file_name="same.docx",
        display_path="docs/same.docx",
        download_path=source_path,
        download_name="same.docx",
        download_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        preview_text="Second preview",
        can_download=True,
    )

    groups = web_app.normalize_retrieved_file_groups([first, second], settings)

    assert len(groups) == 1
    assert groups[0].file_key == "relative_path:docs/same.docx"
    assert groups[0].preview_text == "First preview"


def test_build_retrieved_file_groups_uses_relative_path_for_safe_download(tmp_path):
    documents_root = tmp_path / "documents"
    source_path = documents_root / "forms" / "заявление на увольнение.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"form")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Фрагмент заявления.",
                metadata={
                    "relative_path": "forms/заявление на увольнение.docx",
                    "file_name": "заявление на увольнение.docx",
                    "source_type": "body",
                },
            )
        ],
        settings,
    )

    assert groups[0].download_path == source_path.resolve()
    assert groups[0].download_name == "заявление на увольнение.docx"
    assert groups[0].can_download is True


def test_build_retrieved_file_groups_disables_missing_or_outside_downloads(tmp_path):
    documents_root = tmp_path / "documents"
    documents_root.mkdir()
    outside_path = tmp_path / "outside.docx"
    outside_path.write_bytes(b"outside")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Missing file snippet.",
                metadata={"relative_path": "missing.docx", "file_name": "missing.docx"},
            ),
            Document(
                page_content="Outside file snippet.",
                metadata={"file_path": str(outside_path), "file_name": "outside.docx"},
            ),
        ],
        settings,
    )

    assert [group.can_download for group in groups] == [False, False]
    assert [group.download_path for group in groups] == [None, None]
    assert [group.preview_text for group in groups] == [
        web_app.PREVIEW_UNAVAILABLE_TEXT,
        web_app.PREVIEW_UNAVAILABLE_TEXT,
    ]


def test_build_retrieved_file_groups_returns_empty_list_without_evidence(tmp_path):
    settings = SimpleNamespace(documents_root=tmp_path / "documents")

    assert build_retrieved_file_groups([], settings) == []


def test_main_loads_project_env_before_creating_settings(monkeypatch):
    from imperial_rag import web_app

    calls = []

    env_module = types.ModuleType("imperial_rag.env")
    env_module.load_project_env = lambda: calls.append("env")

    class FakeSettings:
        def __init__(self):
            calls.append("settings")

    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = types.ModuleType("imperial_rag.tracing")
    tracing_module.configure_phoenix_tracing = lambda settings: calls.append("tracing")

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        sidebar=Sidebar(),
        header=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
        session_state=SessionState(),
        chat_input=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.tracing", tracing_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)

    web_app.main()

    assert calls[:2] == ["env", "settings"]


def test_main_bootstraps_src_path_for_streamlit_script_launch():
    script = textwrap.dedent(
        """
        import runpy
        import sys
        import types

        sys.path = [entry for entry in sys.path if not entry.endswith('/src')]
        for name in list(sys.modules):
            if name.startswith('imperial_rag'):
                del sys.modules[name]

        class SessionState(dict):
            def __getattr__(self, key):
                return self[key]

            def __setattr__(self, key, value):
                self[key] = value

        class Sidebar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        sys.modules['streamlit'] = types.SimpleNamespace(
            set_page_config=lambda **kwargs: None,
            title=lambda *args, **kwargs: None,
            sidebar=Sidebar(),
            header=lambda *args, **kwargs: None,
            text=lambda *args, **kwargs: None,
            session_state=SessionState(),
            chat_input=lambda *args, **kwargs: None,
        )
        namespace = runpy.run_path('src/imperial_rag/web_app.py', run_name='imperial_web_app_test')
        namespace['main']()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
