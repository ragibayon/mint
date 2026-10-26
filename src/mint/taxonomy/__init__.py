"""Seeded contract-type taxonomy adaptation and reporting."""

__all__ = [
    "Assignment",
    "Codebook",
    "ContractRecord",
    "ExtensionProposal",
    "load_seed_codebook",
]


def __getattr__(name: str):
    if name in {"Codebook", "load_seed_codebook"}:
        from . import codebook

        return getattr(codebook, name)
    if name in {"Assignment", "ContractRecord", "ExtensionProposal"}:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
