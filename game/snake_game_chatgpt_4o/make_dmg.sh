#!/bin/bash

# Unmount existing DMG
hdiutil detach /Volumes/Snake\ Game 2>/dev/null || true
hdiutil detach /Volumes/dmg.LPjjen 2>/dev/null || true
hdiutil detach /dev/disk4 2>/dev/null || true

# Clean up
rm -rf build dist
find . -name '*.egg-info' -exec rm -rf {} +
#rm -f dist/Snake_Game_Installer.dmg
source venv311/bin/activate
python setup.py py2app

# Create new DMG
create-dmg \
  --volname "Snake Game" \
  --window-pos 200 120 \
  --window-size 500 300 \
  --icon-size 100 \
  --icon "snake_game.app" 100 100 \
  --app-drop-link 350 100 \
  --volicon "snake_icon.icns" \
  dist/Snake_Game_Installer.dmg \
  dist
