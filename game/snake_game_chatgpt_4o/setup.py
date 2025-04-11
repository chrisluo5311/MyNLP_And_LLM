from setuptools import setup

APP = ['snake_game.py']
DATA_FILES = [
    ('sounds', ['sounds/eat.wav', 'sounds/gameover.wav', 'sounds/powerup.wav']),
    ('', ['scores.txt'])  # ensure scores.txt is included
]
OPTIONS = {
    'argv_emulation': False,
    'packages': ['pygame'],
    'iconfile': "snake_icon.icns",
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
