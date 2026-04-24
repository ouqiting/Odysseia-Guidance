import importlib
import sys
import types
from pathlib import Path


def test_personal_memory_search_service_defaults(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    sqlalchemy_stub = types.ModuleType("sqlalchemy")
    sqlalchemy_stub.text = lambda value: value
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_stub)

    database_stub = types.ModuleType("src.database.database")
    database_stub.AsyncSessionLocal = object()
    monkeypatch.setitem(sys.modules, "src.database.database", database_stub)

    gemini_stub_module = types.ModuleType("src.chat.services.gemini_service")
    gemini_stub_module.gemini_service = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "src.chat.services.gemini_service",
        gemini_stub_module,
    )

    module_name = (
        "src.chat.features.personal_memory.services.personal_memory_search_service"
    )
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    service = module.PersonalMemorySearchService()

    assert service.top_k_relevant == 20
    assert service.top_k_random == 5
