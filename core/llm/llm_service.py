"""Сервис для работы с LLM на основе llama-cpp-python (GGUF-модели)."""

import logging
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaService:
    """Сервис для генерации текста с использованием GGUF-модели."""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, verbose: bool = False):
        """
        Инициализация сервиса.
        
        Args:
            model_path: Путь к GGUF-модели.
            n_ctx: Размер контекста.
            verbose: Включить verbose-логи.
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.llm: Optional[Llama] = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели (ленивая)."""
        if not self.model_path.exists():
            logger.error(f"Модель не найдена: {self.model_path}")
            return
        
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                verbose=self.verbose,
                n_threads=4,  # Настраиваемо под систему
            )
            logger.info(f"Модель загружена: {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {self.model_path}: {e}")
            self.llm = None
    
    def generate(
        self,
        system_prompt: str,
        user_query: str,
        context: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Генерация ответа.
        
        Args:
            system_prompt: Системный промпт.
            user_query: Запрос пользователя.
            context: Контекст из retrieval.
            max_tokens: Макс. токенов.
            temperature: Температура.
            
        Returns:
            str: Сгенерированный ответ или None при ошибке.
        """
        if not self.llm:
            logger.error("LLM не загружена.")
            return None
        
        # Формируем полный промпт
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\nContext:\n{context}\n\nQuestion: {user_query}\n<|assistant|>\n"
        
        try:
            response = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|assistant|>", "\n\n"],
                echo=False
            )
            generated_text = response['choices'][0]['text'].strip()
            # Remove potential assistant token if it appears in the response
            if generated_text.startswith("<|assistant|>"):
                generated_text = generated_text[len("<|assistant|>"):].strip()
            logger.info(f"Генерация завершена: {len(generated_text)} символов")
            return generated_text
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return None


# Глобальный экземпляр (синглтон)
_llm_service: Optional[LlamaService] = None


def get_llm_service(config) -> Optional[LlamaService]:
    """Получить сервис LLM (синглтон)."""
    global _llm_service
    if _llm_service is None and config.rag_enabled:
        from config.resource_path import resource_path
        model_path = resource_path(config.rag_model_path)
        _llm_service = LlamaService(str(model_path), config.gguf_model_n_ctx)
    return _llm_service