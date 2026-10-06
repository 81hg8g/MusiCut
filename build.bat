@echo off
cd /d E:\Works\MusiCut

echo === Cleaning old build ===
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo === Building with PyInstaller ===
python -m PyInstaller -y MusiCut.spec --noconfirm
echo.
if exist "dist\MusiCut\MusiCut.exe" (
    echo SUCCESS: dist\MusiCut\MusiCut.exe
    dir dist\MusiCut\MusiCut.exe
) else (
    echo FAILED.
)

pause
