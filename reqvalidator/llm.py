"""Клиент LLM (Google Gemini) с троттлингом под бесплатный тариф, повторами, JSON-схемами и дисковым кэшем."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


@dataclass
class LLMStats:
    calls: int = 0              # реальные обращения к API (успешные)
    cached: int = 0             # ответы, взятые из кэша прежних реальных вызовов
    retries: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    log: list[str] = field(default_factory=list)


class LLM(Protocol):
    model: str
    stats: LLMStats

    def generate(self, system: str, prompt: str, schema: type[T], tag: str = "") -> T: ...


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _clean_json(text: str) -> str:
    return _FENCE.sub("", text.strip())


class GeminiClient:
    def __init__(self, api_key: str, model: str, min_interval: float = 6.0, max_retries: int = 6,
                 cache_dir: Optional[Path] = None, verbose: bool = False,
                 fallback_models: Optional[list[str]] = None, switch_after: int = 2):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMError("Не установлен пакет google-genai (pip install -r requirements.txt)") from e
        self._genai, self._types = genai, types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=180_000))
        self.model = model
        self.primary_model = model                      # для ключа кэша (чтобы кэш не зависел от fallback)
        self.fallback_models = [m for m in (fallback_models or []) if m != model]
        self.switch_after = switch_after
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.stats = LLMStats()
        self._last_call = 0.0
        self._use_schema = True
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ кэш
    def _key(self, system: str, prompt: str, schema: type[BaseModel]) -> str:
        h = hashlib.sha256()
        for part in (self.primary_model, schema.__name__, system, prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    # --------------------------------------------------------------- вызовы
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _retry_delay(err: Exception) -> Optional[float]:
        m = re.search(r"retry in ([\d.]+)\s*s", str(err), re.IGNORECASE) or re.search(r"retryDelay\W+(\d+)s", str(err))
        return float(m.group(1)) + 1 if m else None

    def _switch_model(self, reason: str) -> bool:
        """Переключиться на следующую запасную модель. False — запасных не осталось."""
        if not self.fallback_models:
            return False
        old, self.model = self.model, self.fallback_models.pop(0)
        _log(f"  🔀 {old}: {reason} → переключаюсь на {self.model}")
        return True

    def _request(self, system: str, prompt: str, schema: type[BaseModel]) -> str:
        types = self._types
        fails = 0  # подряд неудачных попыток на текущей модели
        for attempt in range(self.max_retries + 1 + 3 * len(self.fallback_models)):
            self._throttle()
            cfg = dict(system_instruction=system, temperature=0.0, response_mime_type="application/json",
                       max_output_tokens=32768, seed=0)
            if self._use_schema:
                cfg["response_schema"] = schema
            t0 = time.monotonic()
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=types.GenerateContentConfig(**cfg))
            except self._genai.errors.APIError as e:
                self._last_call = time.monotonic()
                code = getattr(e, "code", None)
                msg = str(e)
                if code == 429 and re.search(r"PerDay|per day|daily", msg, re.IGNORECASE):
                    # суточный лимит у каждой модели свой — пробуем другую
                    if self._switch_model("исчерпан суточный лимит"):
                        fails = 0
                        continue
                    raise LLMError(
                        "Исчерпан суточный лимит бесплатного тарифа Gemini. Повторите запуск завтра "
                        "(уже полученные ответы сохранены в кэше и не будут запрашиваться снова) "
                        "или выберите другую модель в GEMINI_MODEL.") from e
                if code in (429, 500, 502, 503, 504):
                    fails += 1
                    self.stats.retries += 1
                    # перегрузка модели: после switch_after неудач берём запасную модель
                    if fails >= self.switch_after and self._switch_model(f"API {code}"):
                        fails = 0
                        continue
                    if fails <= self.max_retries:
                        delay = self._retry_delay(e) or min(90.0, 5.0 * 2 ** (fails - 1))
                        _log(f"  ⏳ API {code} ({self.model}), повтор через {delay:.0f} с "
                             f"(попытка {fails}/{self.max_retries})")
                        time.sleep(delay)
                        continue
                if code == 400 and self._use_schema and re.search(r"schema|response_", msg, re.IGNORECASE):
                    _log("  ⚠ модель не приняла response_schema — продолжаю с JSON-режимом без схемы")
                    self._use_schema = False
                    continue
                if code in (401, 403):
                    raise LLMError(f"Ключ API отклонён ({code}). Проверьте GEMINI_API_KEY в .env") from e
                if code == 404:
                    if self._switch_model("модель не найдена"):
                        fails = 0
                        continue
                    raise LLMError(f"Модель «{self.model}» не найдена. Проверьте GEMINI_MODEL в .env") from e
                raise LLMError(f"Ошибка API Gemini ({code}): {msg[:400]}") from e
            except Exception as e:  # сеть, таймаут
                self._last_call = time.monotonic()
                fails += 1
                if fails <= self.max_retries:
                    delay = min(60.0, 4.0 * 2 ** (fails - 1))
                    self.stats.retries += 1
                    _log(f"  ⏳ сетевая ошибка ({type(e).__name__}), повтор через {delay:.0f} с")
                    time.sleep(delay)
                    continue
                raise LLMError(f"Сетевая ошибка при обращении к Gemini: {e}") from e
            self._last_call = time.monotonic()
            self.stats.seconds += self._last_call - t0
            um = getattr(resp, "usage_metadata", None)
            if um:
                self.stats.prompt_tokens += getattr(um, "prompt_token_count", 0) or 0
                self.stats.output_tokens += getattr(um, "candidates_token_count", 0) or 0
            text = resp.text
            if not text:
                self.stats.retries += 1
                _log("  ⚠ пустой ответ модели, повторяю")
                continue
            return text
        raise LLMError("Модель не вернула ответ после всех повторов")

    def generate(self, system: str, prompt: str, schema: type[T], tag: str = "") -> T: ...


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _clean_json(text: str) -> str:
    return _FENCE.sub("", text.strip())


class GeminiClient:
    def __init__(self, api_key: str, model: str, min_interval: float = 6.0, max_retries: int = 6,
                 cache_dir: Optional[Path] = None, verbose: bool = False,
                 fallback_models: Optional[list[str]] = None, switch_after: int = 2):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMError("Не установлен пакет google-genai (pip install -r requirements.txt)") from e
        self._genai, self._types = genai, types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=180_000))
        self.model = model
        self.primary_model = model                      # для ключа кэша (чтобы кэш не зависел от fallback)
        self.fallback_models = [m for m in (fallback_models or []) if m != model]
        self.switch_after = switch_after
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.stats = LLMStats()
        self._last_call = 0.0
        self._use_schema = True
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ кэш
    def _key(self, system: str, prompt: str, schema: type[BaseModel]) -> str:
        h = hashlib.sha256()
        for part in (self.primary_model, schema.__name__, system, prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    # --------------------------------------------------------------- вызовы
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _retry_delay(err: Exception) -> Optional[float]:
        m = re.search(r"retry in ([\d.]+)\s*s", str(err), re.IGNORECASE) or re.search(r"retryDelay\W+(\d+)s", str(err))
        return float(m.group(1)) + 1 if m else None

    def _request(self, system: str, prompt: str, schema: type[BaseModel]) -> str:
        types = self._types
        for attempt in range(self.max_retries + 1):
            self._throttle()
            cfg = dict(system_instruction=system, temperature=0.0, response_mime_type="application/json",
                       max_output_tokens=32768, seed=0)
            if self._use_schema:
                cfg["response_schema"] = schema
            t0 = time.monotonic()
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=types.GenerateContentConfig(**cfg))
            except self._genai.errors.APIError as e:
                self._last_call = time.monotonic()
                code = getattr(e, "code", None)
                msg = str(e)
                if code == 429 and re.search(r"PerDay|per day|daily", msg, re.IGNORECASE):
                    raise LLMError(
                        "Исчерпан суточный лимит бесплатного тарифа Gemini. Повторите запуск завтра "
                        "(уже полученные ответы сохранены в кэше и не будут запрашиваться снова) "
                        "или выберите другую модель в GEMINI_MODEL.") from e
                if code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    delay = self._retry_delay(e) or min(90.0, 5.0 * 2 ** attempt)
                    self.stats.retries += 1
                    _log(f"  ⏳ API {code}, повтор через {delay:.0f} с (попытка {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue
                if code == 400 and self._use_schema and re.search(r"schema|response_", msg, re.IGNORECASE):
                    _log("  ⚠ модель не приняла response_schema — продолжаю с JSON-режимом без схемы")
                    self._use_schema = False
                    continue
                if code in (401, 403):
                    raise LLMError(f"Ключ API отклонён ({code}). Проверьте GEMINI_API_KEY в .env") from e
                if code == 404:
                    raise LLMError(f"Модель «{self.model}» не найдена. Проверьте GEMINI_MODEL в .env") from e
                raise LLMError(f"Ошибка API Gemini ({code}): {msg[:400]}") from e
            except Exception as e:  # сеть, таймаут
                self._last_call = time.monotonic()
                if attempt < self.max_retries:
                    delay = min(60.0, 4.0 * 2 ** attempt)
                    self.stats.retries += 1
                    _log(f"  ⏳ сетевая ошибка ({type(e).__name__}), повтор через {delay:.0f} с")
                    time.sleep(delay)
                    continue
                raise LLMError(f"Сетевая ошибка при обращении к Gemini: {e}") from e
            self._last_call = time.monotonic()
            self.stats.seconds += self._last_call - t0
            um = getattr(resp, "usage_metadata", None)
            if um:
                self.stats.prompt_tokens += getattr(um, "prompt_token_count", 0) or 0
                self.stats.output_tokens += getattr(um, "candidates_token_count", 0) or 0
            text = resp.text
            if not text:
                self.stats.retries += 1
                _log("  ⚠ пустой ответ модели, повторяю")
                continue
            return text
        raise LLMError("Модель не вернула ответ после всех повторов")

    def generate(self, system: str, prompt: str, schema: type[T], tag: str = "") -> T:
        key = self._key(system, prompt, schema)
        cache_file = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            try:
                out = schema.model_validate_json(cache_file.read_text(encoding="utf-8"))
                self.stats.cached += 1
                if self.verbose:
                    _log(f"  ↺ {tag}: из кэша")
                return out
            except (ValidationError, ValueError):
                cache_file.unlink(missing_ok=True)

        suffix = ""
        last_err: Optional[Exception] = None
        for _ in range(3):
            text = self._request(system, prompt + suffix, schema)
            try:
                out = schema.model_validate_json(_clean_json(text))
            except (ValidationError, ValueError) as e:
                last_err = e
                self.stats.retries += 1
                _log(f"  ⚠ {tag}: ответ не прошёл валидацию схемы, повторяю")
                suffix = ("\n\nВНИМАНИЕ: предыдущий ответ был невалидным JSON или не соответствовал схеме. "
                          "Верни ТОЛЬКО корректный JSON строго по описанной схеме, без пояснений и markdown.")
                continue
            self.stats.calls += 1
            if cache_file:
                cache_file.write_text(out.model_dump_json(), encoding="utf-8")
            return out
        raise LLMError(f"{tag}: модель трижды вернула невалидный JSON: {str(last_err)[:300]}")
