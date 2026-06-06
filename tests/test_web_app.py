import subprocess
import sys
import textwrap
import types

from imperial_rag.web_app import APP_TITLE, build_status_summary, load_status_summary


def test_status_summary_displays_manifest_counts():
    summary = build_status_summary(total_files=162, indexed_files=100, failed_files=3)

    assert APP_TITLE == "Imperial RAG"
    assert "Total files: 162" in summary
    assert "Indexed files: 100" in summary
    assert "Failed files: 3" in summary


def test_load_status_summary_is_importable_without_manifest_stack():
    summary = load_status_summary(settings=object())

    assert "Total files:" in summary


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
