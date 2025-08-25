/**
 * AJAX функции для KAZrag
 */

// Функция для выполнения AJAX POST запроса
async function ajaxPost(url, data) {
    const formData = new FormData();
    for (const key in data) {
        if (data.hasOwnProperty(key)) {
            formData.append(key, data[key]);
        }
    }
    
    const response = await fetch(url, {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
        return await response.json();
    } else {
        return await response.text();
    }
}

// Функция для поиска документов
async function searchDocuments(query, collection, searchDevice, k, searchType, filters) {
    const data = {
        query: query,
        collection: collection,
        search_device: searchDevice,
        k: k,
        search_type: searchType,
        filter_author: filters.author || '',
        filter_source: filters.source || '',
        filter_file_extension: filters.fileExtension || '',
        metadata_filter: filters.custom || ''
    };
    
    // Добавляем hybrid параметр для обратной совместимости
    if (searchType === 'hybrid') {
        data.hybrid = 'True';
    } else {
        data.hybrid = 'False';
    }
    
    return await ajaxPost('/api/search/', data);
}

// Функция для обновления результатов поиска
function updateSearchResults(htmlContent) {
    const resultsContainer = document.getElementById('search-results-container');
    if (resultsContainer) {
        resultsContainer.innerHTML = htmlContent;
    }
}

// Функция для отображения сообщения об ошибке
function showErrorMessage(message) {
    const errorContainer = document.getElementById('error-message');
    if (errorContainer) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';
    }
}

// Функция для скрытия сообщения об ошибке
function hideErrorMessage() {
    const errorContainer = document.getElementById('error-message');
    if (errorContainer) {
        errorContainer.style.display = 'none';
    }
}

// Функция для отображения индикатора загрузки
function showLoadingIndicator() {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
    }
}

// Функция для скрытия индикатора загрузки
function hideLoadingIndicator() {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
}

// Функция для обработки отправки формы поиска
async function handleSearchSubmit(event) {
    event.preventDefault();
    
    // Скрываем сообщения об ошибках
    hideErrorMessage();
    
    // Отображаем индикатор загрузки
    showLoadingIndicator();
    
    // Получаем значения из формы
    const query = document.getElementById('query').value;
    const collection = document.getElementById('collection').value;
    const searchDevice = document.getElementById('search_device').value;
    const searchType = document.querySelector('input[name="search_type"]:checked').value;
    
    // Получаем значения фильтров
    const filters = {
        author: document.getElementById('filter_author').value,
        source: document.getElementById('filter_source').value,
        fileExtension: document.getElementById('filter_file_extension').value,
        custom: document.getElementById('metadata_filter').value
    };
    
    // Получаем значение k из поля ввода или используем значение по умолчанию
    const kInput = document.getElementById('k');
    const k = kInput && kInput.value ? kInput.value : 5; // Значение по умолчанию 5
    
    try {
        // Выполняем поиск
        const response = await searchDocuments(query, collection, searchDevice, k, searchType, filters);
        
        // Обновляем результаты поиска
        updateSearchResults(response);
    } catch (error) {
        console.error('Ошибка при выполнении поиска:', error);
        showErrorMessage('Произошла ошибка при выполнении поиска. Пожалуйста, попробуйте еще раз.');
    } finally {
        // Скрываем индикатор загрузки
        hideLoadingIndicator();
    }
}

// Функция для инициализации обработчиков событий
function initAjaxHandlers() {
    // Обработчик для формы поиска
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        searchForm.addEventListener('submit', handleSearchSubmit);
    }
    
    // Обработчики для радиокнопок поиска
    const searchTypeRadios = document.querySelectorAll('input[name="search_type"]');
    searchTypeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                document.getElementById('hybrid').value = this.value === 'hybrid' ? 'True' : 'False';
            }
        });
    });
    
    // Обработчик для спойлера фильтра по метаданным
    const collapsible = document.querySelector('.collapsible');
    if (collapsible) {
        collapsible.addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            if (content.style.display === 'block') {
                content.style.display = 'none';
            } else {
                content.style.display = 'block';
            }
        });
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    initAjaxHandlers();
});