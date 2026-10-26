from pathlib import Path

from .github_agent import GithubContractNormalizer

PROMPT_VERSION = "huggingface"


class HuggingFaceContractNormalizer(GithubContractNormalizer):
    def __init__(self) -> None:
        super().__init__(Path(__file__).parent / "prompts" / PROMPT_VERSION)
