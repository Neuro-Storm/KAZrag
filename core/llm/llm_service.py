"""Сервис для работы с LLM на основе llama-cpp-python (GGUF-модели)."""

import logging
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LlamaService:
    """Сервис для генерации текста с использованием GGUF-модели."""
    
    def __init__(self, 
                 model_path: str, 
                 n_ctx: int = 4096, 
                 n_gpu_layers: int = -1,
                 n_threads: int = 4,
                 n_batch: int = 512,
                 n_beams: int = 1,
                 verbose: bool = False):
        """
        Инициализация сервиса.
        
        Args:
            model_path: Путь к GGUF-модели.
            n_ctx: Размер контекста.
            n_gpu_layers: Количество слоев для GPU (-1 = все слои)
            n_threads: Количество потоков
            n_batch: Размер батча
            n_beams: Размер beam для генерации
            verbose: Включить verbose-логи.
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_beams = n_beams
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
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                verbose=self.verbose,
            )
            logger.info(f"Модель загружена: {self.model_path} (context={self.n_ctx}, gpu_layers={self.n_gpu_layers}, threads={self.n_threads})")
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
        
        # Проверяем длину промпта и обрезаем контекст при необходимости
        prompt_length = len(full_prompt.encode('utf-8'))
        max_context_bytes = self.n_ctx * 4  # примерное ограничение (4 байта на токен как грубая оценка)
        
        if prompt_length > max_context_bytes:
            logger.warning(f"Длина промпта ({prompt_length}) превышает ограничение ({max_context_bytes}), обрезаем контекст")
            # Обрезаем контекст, оставляя место для промпта и ответа
            base_prompt = f"<|system|>\n{system_prompt}\n<|user|>\nContext:\n\nQuestion: {user_query}\n<|assistant|>\n"
            base_prompt_length = len(base_prompt.encode('utf-8'))
            available_context_bytes = max_context_bytes - base_prompt_length - (max_tokens * 4)
            
            if available_context_bytes > 0:
                # Обрезаем контекст до доступного размера
                truncated_context = context[:available_context_bytes]
                full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\nContext:\n{truncated_context}\n\nQuestion: {user_query}\n<|assistant|>\n"
                logger.info(f"Контекст обрезан до {len(truncated_context)} байт")
            else:
                logger.error("Недостаточно места для контекста даже после обрезки")
                return None
        
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
_llm_service_params: Optional[dict] = None


def get_llm_service(config) -> Optional[LlamaService]:
    """Получить сервис LLM (синглтон)."""
    global _llm_service, _llm_service_params
    
    # Проверяем, нужно ли пересоздать сервис
    current_params = {
        'model_path': config.rag_model_path,
        'n_ctx': config.rag_context_size,
        'n_gpu_layers': config.rag_gpu_layers,
        'n_threads': config.rag_threads,
        'n_batch': config.rag_batch_size,
        'n_beams': config.rag_beam_size
    }
    
    if config.rag_enabled:
        # Если сервис не создан или параметры изменились, создаем новый
        if _llm_service is None or _llm_service_params != current_params:
            from config.resource_path import resource_path
            model_path = resource_path(config.rag_model_path)
            _llm_service = LlamaService(
                str(model_path), 
                n_ctx=config.rag_context_size,
                n_gpu_layers=config.rag_gpu_layers,
                n_threads=config.rag_threads,
                n_batch=config.rag_batch_size,
                n_beams=config.rag_beam_size
            )
            _llm_service_params = current_params
            logger.info(f"LLM сервис пересоздан с новыми параметрами: {current_params}")
    else:
        # Если RAG отключен, сбрасываем сервис
        _llm_service = None
        _llm_service_params = None
    
    return _llm_service