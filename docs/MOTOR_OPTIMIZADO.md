# Motor de Auditoría de Rupturas — Versión Optimizada con Numba

Motor cuántico de auditoría multiescala optimizado con Numba.
Elimina fugas de información y corrige distorsiones de fase espectral.

## Características principales

- ✅ **PCA rodante sin look-ahead bias** (Numba JIT)
- ✅ **Filtro Butterworth** para Hilbert (sin artefactos)
- ✅ **Tolerancia adaptativa QH** (desviación estándar dinámica)
- ✅ **Z-score causal** (historial puro)
- ✅ **Entropía Shannon + Hugo normalizadas**

## Instalación

```bash
pip install numpy pandas scipy scikit-learn numba matplotlib yfinance
```

## Uso Básico

```python
from motor_optimizado import MotorAuditoriaRupturas
import yfinance as yf

# Descargar datos
precios = yf.download("BTC-USD", period="5y")["Close"]

# Crear motor
motor = MotorAuditoriaRupturas(
    horizontes=[5, 10, 21, 63],
    ventana_zscore=252
)

# Ejecutar análisis
features = motor.generar_caracteristicas(precios)
rupturas = motor.detectar_rupturas(features, precios)

# Ver resultados
print(rupturas[['Precio', 'PC1', 'Z_score']].tail())
```

## Correcciones Implementadas

### 1. Filtro Butterworth para Hilbert
Elimina artefactos de borde y fuga espectral en la transformada de Hilbert.

### 2. Tolerancia Adaptativa para QH
Reemplaza umbral fijo por desviación estándar dinámica.

### 3. Numba JIT para PCA
PCA rodante compilado a código máquina (~100x más rápido).

---

**Última actualización**: 2026-08-24
