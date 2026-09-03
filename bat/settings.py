"""Typed, environment-driven configuration.

Nothing here has a hardcoded machine path. The previous ``core/config.py``
pinned storage under the repo directory and ``engine.py`` hardcoded
``D:/BatAI/brain_memory``; both are fatal in a containerised deployment where
the process has no idea where it lives.

All settings are read from the environment with the ``BAT_`` prefix, or from a
``.env`` file in local development only.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bat.domain.tenancy import DEFAULT_SCOPES, Scope, validate_tenant_id
from bat.ports.tools import Authority, Isolation


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        return self is Environment.PRODUCTION


class ApiKeyRecord(BaseModel):
    """A credential grant.

    Only the SHA-256 hash of the key is ever configured or stored; the platform
    never holds a verifiable plaintext credential at rest.
    """

    model_config = {"frozen": True}

    key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_id: str
    principal_id: str
    scopes: frozenset[Scope] = DEFAULT_SCOPES
    label: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def _check_tenant(cls, v: str) -> str:
        return validate_tenant_id(v)

    @staticmethod
    def hash_key(plaintext: str) -> str:
        return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class ModelSettings(BaseModel):
    """Native llama.cpp inference settings.

    Inference runs *inside* this process against a local .gguf file. There is no
    model server, so the model is a process-local resource: one instance serves
    one generation at a time and every worker holds its own full copy of the
    weights in RAM. `max_queue_depth` is the admission bound that keeps that
    honest under load -- see `bat.adapters.llama_cpp_client`.
    """

    model_config = {"frozen": True}

    provider: Literal["llama_cpp"] = "llama_cpp"
    #: Path to the .gguf weights. Required before the API will serve turns.
    model_path: Path | None = None
    #: Display name used in logs and usage records.
    name: str = "bat-engine"

    # -- llama.cpp load parameters ---------------------------------------
    n_ctx: int = Field(default=8192, gt=0)
    #: Layers offloaded to GPU. 0 = pure CPU; -1 = offload everything.
    n_gpu_layers: int = Field(default=0, ge=-1)
    #: Generation threads. None lets llama.cpp pick from the CPU count.
    n_threads: int | None = Field(default=None, gt=0)
    n_batch: int = Field(default=512, gt=0)
    #: Chat template. None uses the template embedded in the .gguf metadata.
    chat_format: str | None = None
    seed: int | None = None
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = False

    # -- generation defaults ---------------------------------------------
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: int = Field(default=2048, gt=0)
    repeat_penalty: float = Field(default=1.1, gt=0)
    request_timeout_s: float = Field(default=180.0, gt=0)

    # -- admission --------------------------------------------------------
    #: Callers allowed to wait for the single generation slot before new ones
    #: are rejected outright. Queueing beyond this only burns client deadlines.
    max_queue_depth: int = Field(default=8, gt=0)
    #: Block startup until the weights are loaded. Off means the first request
    #: pays the load cost and /readyz reports not-ready until it finishes.
    preload: bool = True

    @model_validator(mode="after")
    def _check_paths(self) -> ModelSettings:
        if self.model_path is not None and self.model_path.suffix.lower() != ".gguf":
            raise ValueError(f"model_path must be a .gguf file, got {self.model_path}")
        return self

    @property
    def is_configured(self) -> bool:
        return self.model_path is not None


class EmbeddingSettings(BaseModel):
    """Embedding model for the RAG pipeline.

    A separate .gguf loaded with ``embedding=True``. Kept distinct from the chat
    model because embedding a document while a generation is in flight would
    otherwise contend for the same serialized instance.
    """

    model_config = {"frozen": True}

    model_path: Path | None = None
    n_ctx: int = Field(default=2048, gt=0)
    n_gpu_layers: int = Field(default=0, ge=-1)
    n_threads: int | None = Field(default=None, gt=0)
    n_batch: int = Field(default=512, gt=0)
    #: Embed in batches so a large ingest does not hold the slot indefinitely.
    batch_size: int = Field(default=16, gt=0)
    normalize: bool = True
    verbose: bool = False

    @model_validator(mode="after")
    def _check_paths(self) -> EmbeddingSettings:
        if self.model_path is not None and self.model_path.suffix.lower() != ".gguf":
            raise ValueError(
                f"embedding model_path must be a .gguf file, got {self.model_path}"
            )
        return self

    @property
    def is_configured(self) -> bool:
        return self.model_path is not None


class VectorSettings(BaseModel):
    model_config = {"frozen": True}

    #: ``persistent`` is single-process only. Anything with more than one worker
    #: must use ``http`` against a Chroma server, or the file lock will fight.
    mode: Literal["persistent", "http", "memory"] = "memory"
    path: Path | None = None
    host: str | None = None
    port: int = 8000
    ssl: bool = False
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    default_top_k: int = Field(default=5, gt=0)
    context_token_budget: int = Field(default=1500, gt=0)
    #: Retrieved chunks scoring below this are dropped rather than padding
    #: the prompt with near-irrelevant text at full authority.
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_mode(self) -> VectorSettings:
        if self.mode == "persistent" and self.path is None:
            raise ValueError("vector.path is required when vector.mode='persistent'")
        if self.mode == "http" and not self.host:
            raise ValueError("vector.host is required when vector.mode='http'")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class SessionSettings(BaseModel):
    model_config = {"frozen": True}

    backend: Literal["memory", "redis", "postgres"] = "memory"
    dsn: SecretStr | None = None
    ttl_seconds: int = Field(default=86_400, gt=0)
    max_sessions_per_principal: int = Field(default=200, gt=0)
    history_window: int = Field(default=20, gt=0)
    max_message_chars: int = Field(default=32_000, gt=0)

    @model_validator(mode="after")
    def _check_dsn(self) -> SessionSettings:
        if self.backend in {"redis", "postgres"} and self.dsn is None:
            raise ValueError(f"session.dsn is required for backend={self.backend!r}")
        return self


class RateLimitSettings(BaseModel):
    model_config = {"frozen": True}

    enabled: bool = True
    #: Sustained requests per second per tenant.
    requests_per_second: float = Field(default=5.0, gt=0)
    #: Burst allowance above the sustained rate.
    burst: int = Field(default=20, gt=0)
    #: Concurrent agent runs per tenant. Agent turns are expensive, so they are
    #: capped separately from cheap CRUD calls.
    max_concurrent_runs: int = Field(default=4, gt=0)


class AgentSettings(BaseModel):
    model_config = {"frozen": True}

    max_steps: int = Field(default=6, gt=0, le=32)
    deadline_s: float = Field(default=120.0, gt=0)
    max_tool_calls_per_run: int = Field(default=8, gt=0)
    #: Ceiling on what a tool may reach. HOST is refused in production: a tool
    #: with host reach on shared infrastructure turns prompt injection into
    #: remote code execution.
    max_tool_authority: Authority = Authority.NETWORK
    #: Containment required of tools that run caller-supplied code.
    min_code_isolation: Isolation = Isolation.SUBPROCESS
    #: Tools this deployment opts in, by name. Empty means no tools at all.
    enabled_tools: frozenset[str] = frozenset()


class Settings(BaseSettings):
    """Root configuration object. Construct once, inject everywhere."""

    model_config = SettingsConfigDict(
        env_prefix="BAT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    service_name: str = "bat-api"
    log_level: Annotated[str, Field(pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")] = "INFO"
    json_logs: bool = True
    docs_enabled: bool = True
    cors_origins: tuple[str, ...] = ()
    #: Trusted proxy hop count for client-IP derivation. 0 = trust no XFF header.
    trusted_proxy_hops: int = 0

    api_keys: tuple[ApiKeyRecord, ...] = ()
    model: ModelSettings = ModelSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vector: VectorSettings = VectorSettings()
    session: SessionSettings = SessionSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    agent: AgentSettings = AgentSettings()

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, v: Any) -> Any:
        """Accept a JSON array in the env var as well as native structures."""
        if isinstance(v, str):
            return tuple(json.loads(v))
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return tuple(json.loads(v))
            return tuple(o.strip() for o in v.split(",") if o.strip())
        return v

    @model_validator(mode="after")
    def _harden(self) -> Settings:
        """Fail closed on configurations that are only safe in development."""
        if not self.environment.is_production:
            return self

        problems: list[str] = []
        if not self.api_keys:
            problems.append("no api_keys configured; the API would reject every request")
        if not self.model.is_configured:
            problems.append(
                "model.model_path is unset; the API cannot serve a single agent turn"
            )
        if "*" in self.cors_origins:
            problems.append("cors_origins may not be '*' in production")
        if self.vector.mode == "memory":
            problems.append("vector.mode='memory' loses all tenant memory on restart")
        if self.session.backend == "memory":
            problems.append(
                "session.backend='memory' is per-process; sessions would be lost on "
                "restart and inconsistent across replicas"
            )
        if self.agent.max_tool_authority >= Authority.HOST:
            problems.append(
                "agent.max_tool_authority must be below HOST in production; a tool "
                "that can reach the host gives a tenant control of shared "
                "infrastructure through prompt injection"
            )
        if self.agent.min_code_isolation < Isolation.SUBPROCESS:
            problems.append(
                "agent.min_code_isolation must be SUBPROCESS or stronger in "
                "production; running caller-supplied code in the API process is "
                "arbitrary code execution"
            )
        if problems:
            raise ValueError(
                "unsafe production configuration:\n  - " + "\n  - ".join(problems)
            )
        return self

    @property
    def openapi_url(self) -> str | None:
        return "/openapi.json" if self.docs_enabled else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Cache is cleared in tests."""
    return Settings()
