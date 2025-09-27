#!/usr/bin/env python3
# Script to fix the execute_search method in search_executor.py
# This will recreate the method with the correct structure.

# Read the original file
with open('D:\\Scripts\\RAG\\KAZrag\\core\\search\\search_executor.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the correct execute_search method
correct_execute_search = """    @staticmethod
    async def execute_search(
        client, 
        search_mode: str,
        vector_name: Optional[str],
        sparse_params: Optional[Dict],
        query: str, 
        k: int, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Tuple[Any, float]], Optional[str]]:
        \"\"\"
        Выполняет поиск с опциональной фильтрацией по метаданным.
        
        Args:
            client: QdrantClient.
            search_mode (str): Режим поиска (\"dense\", \"sparse\", \"hybrid\").
            vector_name (str): Имя dense vector.
            sparse_params (Dict): Параметры sparse.
            query (str): Поисковый запрос.
            k (int): Количество результатов.
            metadata_filter (Optional[Dict[str, Any]]): Фильтр по метаданным.
            
        Returns:
            Tuple[List[Tuple[Any, float]], Optional[str]]: (результаты поиска, ошибка)
        \"\"\"
        try:
            if metadata_filter:
                # Создаем фильтр для Qdrant
                must_conditions = []
                
                for key, value in metadata_filter.items():
                    # Обрабатываем различные типы условий
                    if isinstance(value, dict):
                        # Сложные условия (например, {\"$gt\": 2020})
                        for op, op_value in value.items():
                            if op == \"$gt\":
                                must_conditions.append(FieldCondition(
                                    key=f\"metadata.{key}\",
                                    range=Range(gt=op_value)
                                ))
                            elif op == \"$gte\":
                                must_conditions.append(FieldCondition(
                                    key=f\"metadata.{key}\",
                                    range=Range(gte=op_value)
                                ))
                            elif op == \"$lt\":
                                must_conditions.append(FieldCondition(
                                    key=f\"metadata.{key}\",
                                    range=Range(lt=op_value)
                                ))
                            elif op == \"$lte\":
                                must_conditions.append(FieldCondition(
                                    key=f\"metadata.{key}\",
                                    range=Range(lte=op_value)
                                ))
                            elif op == \"$contains\":
                                # Для массивов или строк
                                must_conditions.append(FieldCondition(
                                    key=f\"metadata.{key}\",
                                    match=MatchValue(value=op_value)
                                ))
                    else:
                        # Простое равенство
                        must_conditions.append(FieldCondition(
                            key=f\"metadata.{key}\",
                            match=MatchValue(value=value)
                        ))
                
                search_filter = Filter(must=must_conditions)
                # Выполняем поиск в зависимости от режима
                config = ConfigManager.get_instance().get()
                embedder = get_dense_embedder(config, \"auto\")
                query_vector = embedder.embed_query(query)
                
                if search_mode == \"hybrid\":
                    results, error = await SearchExecutor.execute_hybrid_search(client, query, embedder, sparse_params.get(\"sparse_embedding\"), k, metadata_filter)
                    if error:
                        return [], error
                elif search_mode == \"sparse\":
                        # Sparse поиск
                        # Для sparse векторов используем именованный sparse вектор
                        sparse_vector_name = sparse_params[\"sparse_vector_name\"] if sparse_params else config.sparse_vector_name
                        # Получаем sparse вектор через sparse embedding
                        sparse_embedding = sparse_params.get(\"sparse_embedding\")
                        sparse_vector = None
                        if sparse_embedding:
                            # sparse_result уже в формате {\"indices\": [...], \"values\": [...]} от адаптера
                            sparse_result = sparse_embedding.embed_query(query)
                            sparse_vector = sparse_result  # уже в правильном формате
                        
                        if sparse_vector:
                            results = client.search(
                                collection_name=config.collection_name,
                                query_vector=None,  # Для sparse поиска query_vector=None
                                sparse_vector={sparse_vector_name: sparse_vector},  # {indices: [...], values: [...]}
                                vector_name=sparse_vector_name,  # Имя sparse named vector
                                limit=k,
                                query_filter=search_filter,
                                with_payload=True,
                                with_vectors=False
                            )
                        else:
                            # Если sparse вектор не создан, возвращаем пустой результат
                            results = []
                else:
                    # Dense
                    # Для dense векторов используем именованный вектор
                    if vector_name:
                        results = client.search(
                            collection_name=config.collection_name,
                            query_vector=(vector_name, query_vector),
                            limit=k,
                            query_filter=search_filter,
                            with_payload=True,
                            with_vectors=False
                        )
                    else:
                        # Если имя вектора не указано, используем имя по умолчанию \"dense_vector\"
                        results = client.search(
                            collection_name=config.collection_name,
                            query_vector=(\"dense_vector\", query_vector),
                            limit=k,
                            query_filter=search_filter,
                            with_payload=True,
                            with_vectors=False
                        )
            else:
                # Аналогично без фильтра
                config = ConfigManager.get_instance().get()
                embedder = get_dense_embedder(config, \"auto\")
                query_vector = embedder.embed_query(query)
                
                if search_mode == \"hybrid\":
                    results, error = await SearchExecutor.execute_hybrid_search(client, query, embedder, sparse_params.get(\"sparse_embedding\"), k, metadata_filter)
                    if error:
                        return [], error
                elif search_mode == \"sparse\":
                        # Sparse поиск
                        # Для sparse векторов используем именованный sparse вектор
                        sparse_vector_name = sparse_params[\"sparse_vector_name\"] if sparse_params else config.sparse_vector_name
                        # Получаем sparse вектор через sparse embedding
                        sparse_embedding = sparse_params.get(\"sparse_embedding\")
                        sparse_vector = None
                        if sparse_embedding:
                            # sparse_result уже в формате {\"indices\": [...], \"values\": [...]} от адаптера
                            sparse_result = sparse_embedding.embed_query(query)
                            sparse_vector = sparse_result  # уже в правильном формате
                        
                        if sparse_vector:
                            results = client.search(
                                collection_name=config.collection_name,
                                query_vector=None,  # Для sparse поиска query_vector=None
                                sparse_vector={sparse_vector_name: sparse_vector},  # {indices: [...], values: [...]}
                                vector_name=sparse_vector_name,  # Имя sparse named vector
                                limit=k,
                                with_payload=True,
                                with_vectors=False
                            )
                        else:
                            # Если sparse вектор не создан, возвращаем пустой результат
                            results = []
                else:
                    # Dense
                    # Для dense векторов используем именованный вектор
                    if vector_name:
                        results = client.search(
                            collection_name=config.collection_name,
                            query_vector=(vector_name, query_vector),
                            limit=k,
                            with_payload=True,
                            with_vectors=False
                        )
                    else:
                        # Если имя вектора не указано, используем имя по умолчанию \"dense_vector\"
                        results = client.search(
                            collection_name=config.collection_name,
                            query_vector=(\"dense_vector\", query_vector),
                            limit=k,
                            with_payload=True,
                            with_vectors=False
                        )
        
            # Обрабатываем результаты для извлечения содержимого чанков
            processed_results = []
            for point in results:  # Теперь results - это список PointStruct
                # Извлекаем дополнительную информацию из payload
                payload = point.payload if hasattr(point, 'payload') else {}
                content = payload.get('content', '') or payload.get('page_content', '')
                metadata = payload.get('metadata', {})
                
                # Создаем расширенный объект результата
                extended_result = {
                    'content': content if content is not None else '',
                    'metadata': metadata,
                    'original_score': point.score if hasattr(point, 'score') else 0  # Сохраняем оригинальную оценку
                }
                
                # Если это многоуровневый чанк, добавляем информацию о микро-чанках
                if 'micro_contents' in metadata:
                    extended_result['micro_contents'] = metadata['micro_contents']
                elif 'micro_contents' in payload:
                    extended_result['micro_contents'] = payload['micro_contents']
                    
                # Добавляем source если есть
                if 'source' in metadata:
                    extended_result['source'] = metadata['source']
                elif 'source' in payload:
                    extended_result['source'] = payload['source']
                elif 'source' not in extended_result and 'source' in metadata:
                    extended_result['source'] = metadata.get('source', '')
                    
                processed_results.append((extended_result, point.score if hasattr(point, 'score') else 0))
                
            logger.debug(f\"Search returned {len(processed_results)} results\")  # Оставить
            
            # Log first result before returning to see what we have
            if processed_results:
                first_result, first_score = processed_results[0]
                logger.debug(f\"Before returning - First result score: {first_score}, keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'not dict'}\")
                if isinstance(first_result, dict):
                    logger.debug(f\"Before returning - original_score: {first_result.get('original_score')}\")
        
            return processed_results, None
            
        except Exception as e:
            logger.exception(f\"Ошибка при поиске: {e}\")
            return [], str(e)
\"\"\"


# Find the start and end of the execute_search method in the original content
start_marker = '@staticmethod\\n    async def execute_search('
end_marker = '    @staticmethod\\n    def _create_filter'

# Extract the parts before and after the execute_search method
before_method = content.split(start_marker)[0]
after_method = content.split(end_marker)[1] if end_marker in content else ''

# Construct the new content
new_content = before_method + start_marker + correct_execute_search + '    ' + end_marker + after_method

# Write the corrected file
with open('D:\\Scripts\\RAG\\KAZrag\\core\\search\\search_executor.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(\"execute_search method has been corrected with proper structure\")