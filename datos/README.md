# 📊 Datos de Ejemplo

> Datasets de ejemplo y pruebas para H2O AutoML.

## 📋 Estructura de Datos

### Matrículas (`matriculas.csv`)
```
fecha,carrera,nivel,cantidad
2023-01-01,Ingeniería,Pregrado,150
2023-01-01,Medicina,Postgrado,45
...
```

### Ventas (`ventas_simuladas.csv`)
```
fecha,producto,region,ventas,precio
2023-01-01,ProductoA,Norte,100,15.99
2023-01-01,ProductoB,Sur,75,24.99
...
```

## 🔍 Uso de Datos

1. **Entrenamiento**
   - Use `matriculas.csv` para predicción de matrículas
   - Use `ventas_simuladas.csv` para pruebas de regresión

2. **Validación**
   - 70% entrenamiento
   - 30% validación

## ⚠️ Limitaciones

- Máximo 100MB por archivo
- Solo formatos CSV y XLSX
- Encoding UTF-8 requerido
- Separador por coma (,)

## 🔒 Seguridad

- No incluir datos sensibles
- No subir datos de producción
- Usar solo para pruebas
