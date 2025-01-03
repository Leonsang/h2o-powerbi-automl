@echo off
setlocal enabledelayedexpansion

echo [94m🚀 Iniciando instalación de H2O AutoML para Power BI...[0m

:: Verificar Python
echo.
echo [94m📋 Verificando requisitos...[0m
python --version >nul 2>&1
if errorlevel 1 (
    echo [91m❌ Python 3 no encontrado. Por favor, instálalo primero.[0m
    exit /b 1
)

:: Verificar Java
java -version >nul 2>&1
if errorlevel 1 (
    echo [91m❌ Java no encontrado. Por favor, instala Java 8 o superior.[0m
    exit /b 1
)

:: Crear entorno virtual
echo.
echo [94m🔧 Creando entorno virtual...[0m
python -m venv .venv
call .venv\Scripts\activate.bat

:: Actualizar pip
echo.
echo [94m📦 Actualizando pip...[0m
python -m pip install --upgrade pip

:: Instalar dependencias
echo.
echo [94m📚 Instalando dependencias...[0m
pip install -r requirements.txt

:: Instalar paquete en modo desarrollo
echo.
echo [94m🔨 Instalando H2O PowerBI...[0m
pip install -e .

:: Crear estructura de directorios
echo.
echo [94m📁 Creando estructura de directorios...[0m
python config\crear_directorios.py

:: Ejecutar tests
echo.
echo [94m🧪 Ejecutando tests...[0m
python -m pytest tests/

:: Verificar instalación
echo.
echo [94m✅ Verificando instalación...[0m
python -c "import h2o; print('H2O versión:', h2o.__version__)"

echo.
echo [92m✨ Instalación completada exitosamente![0m
echo.
echo [94m📝 Para usar en Power BI, configura el intérprete de Python en:[0m
echo    %CD%\.venv\Scripts\python.exe

:: Instrucciones finales
echo.
echo [94m📚 Documentación disponible en:[0m
echo    %CD%\docs\documentacion.md

pause 