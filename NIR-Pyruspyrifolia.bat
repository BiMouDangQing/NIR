@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

rem ============================================================
rem  秋月梨 NIR 近红外光谱分析系统 - 启动脚本
rem  Qiuyue Pear NIR Spectrum Analysis System - Launcher
rem ------------------------------------------------------------
rem  所有路径均基于本脚本所在目录动态计算；整个项目目录（含内置
rem  runtime 运行时）可整体拷贝到其它机器离线运行，无需安装 Python。
rem  All paths are relative to this script's own folder; the whole
rem  project folder (including the bundled "runtime") can be copied
rem  to another machine and run offline, no Python install needed.
rem  启动后立即关闭命令窗口，界面由无窗口的 pythonw 进程承载。
rem  This console closes right after launch; the GUI runs on pythonw.
rem ============================================================

set "PROJECT_DIR=%~dp0"
set "PYTHON=%PROJECT_DIR%runtime\python.exe"
set "PYTHONW=%PROJECT_DIR%runtime\pythonw.exe"
set "APP_FILE=%PROJECT_DIR%QT\app.py"

if not exist "%PYTHONW%" (
    echo [ERROR] 未找到内置运行时 / Bundled runtime not found:
    echo         %PYTHONW%
    echo.
    echo 请完整拷贝项目目录（必须包含 runtime 目录）。
    echo Please copy the COMPLETE project folder, including the "runtime" directory.
    echo.
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo [ERROR] 未找到入口文件 / Entry file not found:
    echo         %APP_FILE%
    echo.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

rem 无窗口启动，并立即退出本脚本（关闭命令窗口）
rem Launch windowless and exit immediately to close this console.
start "" "%PYTHONW%" -m QT.app %*
exit /b 0
