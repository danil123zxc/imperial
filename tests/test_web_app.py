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
