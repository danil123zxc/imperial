from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_phoenix_retrieval_judge(monkeypatch):
    """Keep runner/dataset contract tests offline; judge integration has its own tests."""
    from imperial_rag.evals import phoenix_experiment

    async def evaluate(example, output):
        return {"status": "completed", "document_count": len(output.get("ranked_documents", [])),
                "span_id": "test-span"}

    monkeypatch.setattr(phoenix_experiment, "PhoenixRetrievalJudge",
                        lambda *a, **kw: SimpleNamespace(evaluate=evaluate, metadata={}))
