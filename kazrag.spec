# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

# Определяем корневую директорию проекта
project_root = Path.cwd()

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Добавляем директории с шаблонами и статикой
        (str(project_root / 'web' / 'templates'), 'web/templates'),
        (str(project_root / 'web' / 'static'), 'web/static'),
        # Добавляем конфигурационные файлы
        (str(project_root / 'config'), 'config'),
        # Добавляем .env.example как пример
        (str(project_root / '.env.example'), '.'),
        # Добавляем README файлы
        (str(project_root / 'README.md'), '.'),
        (str(project_root / 'LICENSE'), '.'),
        # Добавляем main.py
        (str(project_root / 'main.py'), '.'),
    ],
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
    name='kazrag',
)