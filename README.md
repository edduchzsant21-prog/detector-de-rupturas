# 🔍 Detector de Rupturas — Motor de Auditoría Walk-Forward

Un motor de análisis de series temporales financieras que detecta **rupturas significativas** en precios usando **PCA (Principal Component Analysis)** con enfoque **walk-forward** para evitar look-ahead bias.

## 🎯 Características Principales

- ✅ **Detección sin look-ahead bias**: PCA y scaler se recalibran en ventana rodante usando solo datos pasados
- ✅ **Análisis de entropía**: Métricas Shannon y Hugo pre/post evento
- ✅ **Z-score adaptativo**: Basado en ventana móvil del PC1
- ✅ **Filtrado de eventos**: Agrupa eventos cercanos para evitar falsos positivos
- ✅ **Visualización integrada**: Gráficos de precio, PC1 y Z-score con eventos marcados

## 📊 Caso de Uso

Ideal para:
- Análisis de volatilidad en criptomonedas (Bitcoin, Ethereum, etc.)
- Detección de cambios de régimen en mercados
- Auditoría de eventos extremos pre/post ruptura
- Backtesting de estrategias cuantitativas

## 🚀 Inicio Rápido

### Instalación

```bash
pip install yfinance scikit-learn scipy matplotlib pandas numpy
```

### Ejemplo Básico

```python
from motor_rupturas import MotorAuditoriaRupturasWF
import yfinance as yf

# Descargar datos
ticker = "BTC-USD"
datos = yf.download(ticker, start="2018-01-01", end="2026-08-23")
precios = datos["Close"].squeeze()

# Inicializar motor
motor = MotorAuditoriaRupturasWF(
    horizontes=[5, 10, 21, 63],
    ventana_zscore=252,
    ventana_calibracion=504,
    paso_recalibracion=21
)

# Ejecutar análisis
features, resultados, auditoria = motor.ejecutar(
    precios=precios,
    umbral=2.58,
    ventana_dias=21,
    dias_exclusion=10
)

# Visualizar
motor.graficar_resultados(resultados, auditoria, ticker=ticker)

# Exportar resultados
auditoria.to_csv("eventos_detectados.csv")
```

## 📚 Documentación Completa

- [Guía de Parámetros](docs/PARAMETROS.md)
- [Arquitectura](docs/ARQUITECTURA.md)
- [Metodología](docs/METODOLOGIA.md)
- [Ejemplos Avanzados](docs/EJEMPLOS.md)

## 🔧 Parámetros Principales

| Parámetro | Descripción | Valor Defecto |
|-----------|-------------|---------------|
| `horizontes` | Períodos para calcular retornos/volatilidad | `[5, 10, 21, 63]` |
| `ventana_zscore` | Días para z-score móvil | `252` |
| `ventana_calibracion` | Días para calibrar PCA/scaler (~2 años) | `504` |
| `paso_recalibracion` | Frecuencia de recalibración | `21` |
| `umbral` | Z-score para detectar rupturas | `2.58` |
| `ventana_dias` | Días pre/post para auditar | `21` |

## 📈 Salida Principal

### `auditoria` DataFrame

```
Fecha_evento          Precio_evento  Z_score  Direccion              Delta_Hugo  Retorno_post
2021-11-09 00:00:00       68789.14     2.89   Ruptura positiva          0.045        0.0234
2022-06-13 00:00:00       19235.67    -2.74   Ruptura negativa         -0.038       -0.0156
...
```

Columnas principales:
- **Precio_evento**: Precio en la fecha de ruptura
- **PC1**: Valor del componente principal en la ruptura
- **Z_score**: Z-score normalizado
- **Dirección**: Tipo de ruptura (positiva/negativa)
- **Shannon_pre/post**: Entropía Shannon antes/después
- **Hugo_pre/post**: Entropía Hugo (normalizada)
- **Delta_Hugo**: Cambio en entropía post-evento
- **Delta_volatilidad**: Cambio en volatilidad
- **Retorno_post**: Retorno en los 21 días posteriores

## 🧮 Metodología Walk-Forward

### Diferencia vs. Enfoque Tradicional

**Enfoque con Look-Ahead Bias (❌ Incorrecto)**:
```
fit_transform(TODA la serie) → PC1 usa info futura
```

**Enfoque Walk-Forward (✅ Correcto)**:
```
Para cada día t:
  1. Calibrar PCA/Scaler con datos [t-504:t] (solo pasado)
  2. Aplicar transform() al dato t (sin re-fit)
  3. Obtener PC1[t] sin sesgos
```

Esto es más lento pero **refleja la realidad del trading**.

## 🎓 Interpretación de Resultados

### ¿Qué significa un evento auditado?

1. **Z-score > 2.58**: PC1 está a más de 2.58 desviaciones estándar
2. **Dirección**: Ruptura al alza (positiva) o a la baja (negativa)
3. **Delta_Hugo**: ¿Cambió el orden/caos en el mercado post-evento?
   - Positivo = Más orden/estructura post-ruptura
   - Negativo = Más caos post-ruptura

4. **Retorno_post**: ¿Qué pasó después?
   - Combina con Delta_Hugo para validar la calidad del evento

## 🔬 Testing y Validación

Ejecutar ambas versiones (walk-forward vs. tradicional):
1. ¿Cuántos eventos detecta cada una?
2. ¿Cambia el Delta_Hugo en significancia?
3. ¿Persisten resultados con diferentes parámetros?

Ver [Testing](docs/TESTING.md) para detalles.

## 📁 Estructura del Proyecto

```
detector-de-rupturas/
├── README.md                          # Este archivo
├── motor_rupturas.py                  # Clase principal
├── examples/
│   ├── basico.py                      # Ejemplo simple
│   ├── multiples_activos.py           # Análisis de varios tickers
│   └── backtesting.py                 # Backtesting de estrategia
├── docs/
│   ├── PARAMETROS.md                  # Referencia de parámetros
│   ├── ARQUITECTURA.md                # Diseño de la clase
│   ├── METODOLOGIA.md                 # Detalles técnicos
│   ├── EJEMPLOS.md                    # Ejemplos avanzados
│   └── TESTING.md                     # Estrategia de testing
└── tests/
    ├── test_features.py               # Tests de características
    ├── test_walkforward.py            # Tests de walk-forward
    └── test_auditoria.py              # Tests de auditoría
```

## ⚙️ Sensibilidad de Parámetros

Explorar estos parámetros para diferentes activos:

```python
# Corto plazo (alta sensibilidad)
motor = MotorAuditoriaRupturasWF(
    ventana_calibracion=252,    # 1 año
    paso_recalibracion=5        # Recalibrar cada 5 días
)

# Largo plazo (baja sensibilidad)
motor = MotorAuditoriaRupturasWF(
    ventana_calibracion=756,    # 3 años
    paso_recalibracion=63       # Recalibrar cada 3 meses
)
```

## 📊 Visualización

La función `graficar_resultados()` genera 3 subgráficos:

1. **Panel Superior**: Precio con eventos marcados
2. **Panel Central**: PC1 (componente principal)
3. **Panel Inferior**: Z-score con umbrales ±2.58

## 🛠️ Dependencias

```
yfinance>=0.2.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
numpy>=1.21.0
```

## 📝 Licencia

MIT License - Ver LICENSE para detalles

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repo
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Add mejora'`)
4. Push a rama (`git push origin feature/mejora`)
5. Abre Pull Request

## 📧 Contacto

Para preguntas o sugerencias: [eddu.chz.sant.21@gmail.com]

---

**Última actualización**: 2026-08-23  
**Versión**: 1.0.0 (Walk-Forward)
