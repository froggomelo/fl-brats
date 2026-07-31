__all__ = ["create_model", "model_report"]


def __getattr__(name: str):
    if name in __all__:
        from .factory import create_model, model_report

        return {"create_model": create_model, "model_report": model_report}[name]
    raise AttributeError(name)
