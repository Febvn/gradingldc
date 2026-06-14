# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Pastikan folder ini ter-copy ke dalam executable/folder build
datas = [
    ('webapp/templates', 'webapp/templates'),
    ('webapp/static', 'webapp/static'),
    ('webapp/sample_images', 'webapp/sample_images'),
    ('src', 'src'),
]

# Tambahkan model jika ada
if os.path.exists('models'):
    datas.append(('models', 'models'))

# Import yang mungkin tidak terdeteksi otomatis oleh PyInstaller
hiddenimports = [
    'flaskwebgui',
    'flask',
    'cv2',
    'numpy',
    'skimage.filters',
    'skimage.exposure',
    'skimage.morphology',
    'skimage.measure',
    'skimage.color',
    'sklearn.cluster',
] + collect_submodules('tensorflow')

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorboard', 'IPython', 'notebook', 'jedi', 'matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SmartGradingKopi',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Biarkan True dulu untuk debugging. Ubah ke False untuk produksi.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None # Bisa tambahkan icon.ico nanti
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SmartGradingKopi',
)
