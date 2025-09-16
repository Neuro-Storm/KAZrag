# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

# Определяем корневую директорию проекта
project_root = Path.cwd()

# Собираем данные из каталогов templates и static
web_datas = []
web_datas += collect_data_files('web', include_py_files=False, subdir='templates')
web_datas += collect_data_files('web', include_py_files=False, subdir='static')

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Добавляем директории с шаблонами и статикой
        (r'D:\Scripts\RAG\KAZrag\web\templates', 'web/templates'),
        (r'D:\Scripts\RAG\KAZrag\web\static', 'web/static'),
        # Добавляем конфигурационные файлы
        (r'D:\Scripts\RAG\KAZrag\config', 'config'),
        # Добавляем .env.example как пример
        (r'D:\Scripts\RAG\KAZrag\.env.example', '.'),
        # Добавляем README файлы
        (r'D:\Scripts\RAG\KAZrag\README.md', '.'),
        (r'D:\Scripts\RAG\KAZrag\LICENSE', '.'),
        # Добавляем main.py
        (r'D:\Scripts\RAG\KAZrag\main.py', '.'),
    ] + web_datas, # Добавляем данные из web
    hiddenimports=[
        'web.search_app',
        'web.admin_app',
        'config.settings',
        'config.config_manager',
        'app.app_factory',
        'app.routes',
        'app.startup',
        'unittest',
        'unittest.mock',
        'asgi_request_id',
        'cachetools',
        'loguru',
        'pydantic_settings',
        'aiohttp',
        'llama_cpp',
        'langchain_text_splitters',
        'starlette',
        'unstructured',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='kazrag',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='kazrag_new',
)