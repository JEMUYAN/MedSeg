import importlib


def __getattr__(name):
    if name == "RAGSystem":
        return importlib.import_module(".rag_system", __package__).RAGSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RAGSystem"]
