import os
from dataclasses import dataclass
from dotenv import load_dotenv

import truststore

# Use the OS trust store (macOS keychain) so the corporate Zscaler root CA
# is trusted, just like it is in the browser. Must run before any requests.
truststore.inject_into_ssl()


load_dotenv()


@dataclass(frozen=True)
class LLMProviderConfig:
    api_key: str
    base_url: str
    model: str


def get_llm_provider_config() -> LLMProviderConfig:
    return LLMProviderConfig(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url=os.getenv("GOOGLE_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
    )
