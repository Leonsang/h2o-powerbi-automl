#!/bin/bash

# Colores para mensajes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Iniciando instalación de H2O AutoML para Power BI...${NC}"

# Verificar Python
echo -e "\n${BLUE}📋 Verificando requisitos...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 no encontrado. Por favor, instálalo primero.${NC}"
    exit 1
fi

# Verificar Java
if ! command -v java &> /dev/null; then
    echo -e "${RED}❌ Java no encontrado. Por favor, instala Java 8 o superior.${NC}"
    exit 1
fi

# Crear entorno virtual
echo -e "\n${BLUE}🔧 Creando entorno virtual...${NC}"
python3 -m venv .venv
source .venv/bin/activate

# Actualizar pip
echo -e "\n${BLUE}📦 Actualizando pip...${NC}"
python -m pip install --upgrade pip

# Instalar dependencias
echo -e "\n${BLUE}📚 Instalando dependencias...${NC}"
pip install -r requirements.txt

# Instalar paquete en modo desarrollo
echo -e "\n${BLUE}🔨 Instalando H2O PowerBI...${NC}"
pip install -e .

# Crear estructura de directorios
echo -e "\n${BLUE}📁 Creando estructura de directorios...${NC}"
python config/crear_directorios.py

# Ejecutar tests
echo -e "\n${BLUE}🧪 Ejecutando tests...${NC}"
python -m pytest tests/

# Verificar instalación
echo -e "\n${BLUE}✅ Verificando instalación...${NC}"
python -c "import h2o; print('H2O versión:', h2o.__version__)"

echo -e "\n${GREEN}✨ Instalación completada exitosamente!${NC}"
echo -e "\n${BLUE}📝 Para usar en Power BI, configura el intérprete de Python en:${NC}"
echo -e "   $(pwd)/.venv/bin/python"

# Instrucciones finales
echo -e "\n${BLUE}📚 Documentación disponible en:${NC}"
echo -e "   $(pwd)/docs/documentacion.md" 