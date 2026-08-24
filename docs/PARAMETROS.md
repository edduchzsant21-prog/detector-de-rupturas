# 📋 Guía de Parámetros

Documentación detallada de todos los parámetros del Motor de Auditoría de Rupturas.

## Constructor: `MotorAuditoriaRupturasWF()`

### Parámetros de Inicialización

#### `horizontes` (list, default: `[5, 10, 21, 63]`)
**Descripción**: Períodos en días para calcular características técnicas.

```python
motor = MotorAuditoriaRupturasWF(horizontes=[5, 10, 21, 63])
```

- Para **cada horizonte** se calcula:
  - `ret_{h}`: Retorno logarítmico acumulado en h días
  - `vol_{h}`: Volatilidad (desv. estándar) en h días

| Horizonte | Uso | Interpretación |
|-----------|-----|----------------|
| 5 | Corto plazo | Tendencia de 1 semana |
| 10 | Medio-corto | Tendencia de 2 semanas |
| 21 | Medio | Tendencia de 1 mes |
| 63 | Largo | Tendencia de ~3 meses |

**Recomendación**: Mantener 4 horizontes para balance óptimo.

---

#### `ventana_zscore` (int, default: `252`)
**Descripción**: Número de días para calcular la media móvil y desviación estándar del PC1.

- `252` = 1 año de trading (~252 días hábiles)

| Valor | Período | Volatilidad |
|-------|---------|-------------|
| 63 | ~3 meses | Muy sensible |
| 126 | ~6 meses | Sensible |
| 252 | 1 año | Recomendado |
| 504 | 2 años | Menos sensible |

**Recomendación**: `252` para Bitcoin/criptos.

---

#### `ventana_calibracion` (int, default: `504`)
**Descripción**: Número de días históricos para calibrar el PCA y StandardScaler.

- **504** = ~2 años de datos históricos

| Valor | Período | Comportamiento |
|-------|---------|----------------|
| 252 | 1 año | Adaptativo |
| 504 | 2 años | Recomendado |
| 756 | 3 años | Conservador |

**Recomendación**: Criptos = `252`, Acciones = `504`, Índices = `756`.

---

#### `paso_recalibracion` (int, default: `21`)
**Descripción**: Frecuencia (en días) de recalibración del PCA y StandardScaler.

| Valor | Frecuencia | Trade-off |
|-------|-----------|----------|
| 1 | Cada día | Preciso pero LENTO |
| 5 | Cada semana | Balance |
| 21 | Cada mes | Recomendado |
| 63 | Cada trimestre | Robusto |

**Recomendación**: `21` para la mayoría de casos.

---

## Método: `ejecutar()`

```python
features, resultados, auditoria = motor.ejecutar(
    precios,
    umbral=2.58,
    ventana_dias=21,
    dias_exclusion=10
)
```

### Parámetros

#### `precios` (pd.Series, requerido)
Series de precios de cierre.

```python
import yfinance as yf
datos = yf.download("BTC-USD", start="2020-01-01")
precios = datos["Close"]
```

#### `umbral` (float, default: `2.58`)
Z-score mínimo para detectar ruptura.

| Umbral | Confianza | Eventos/año |
|--------|-----------|------------|
| 1.96 | 95% | ~10 |
| 2.33 | 99% | ~3-5 |
| 2.58 | 99.5% | ~2-3 (✅) |
| 3.00 | 99.73% | ~1-2 |

**Recomendación**: `2.58` para balance.

#### `ventana_dias` (int, default: `21`)
Días antes/después del evento para auditar.

| Valor | Análisis |
|-------|----------|
| 5 | Muy corto |
| 10 | Corto |
| 21 | Balance (✅) |
| 63 | Largo |

#### `dias_exclusion` (int, default: `10`)
Días mínimos entre eventos para evitar duplicados.

| Valor | Efecto |
|-------|--------|
| 5 | Más detalle |
| 10 | Balance (✅) |
| 21 | Agrupa eventos |

---

## Recomendaciones por Activo

### 🪙 Bitcoin/Criptomonedas

```python
motor = MotorAuditoriaRupturasWF(
    horizontes=[5, 10, 21, 63],
    ventana_zscore=126,
    ventana_calibracion=252,
    paso_recalibracion=5
)
```

### 📈 Acciones

```python
motor = MotorAuditoriaRupturasWF(
    horizontes=[10, 21, 63, 252],
    ventana_zscore=252,
    ventana_calibracion=504,
    paso_recalibracion=21
)
```

### 📊 Índices

```python
motor = MotorAuditoriaRupturasWF(
    horizontes=[21, 63, 252],
    ventana_zscore=504,
    ventana_calibracion=756,
    paso_recalibracion=63
)
```

---

**Última actualización**: 2026-08-23
