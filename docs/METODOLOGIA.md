# 🧮 Metodología Técnica

Explicación detallada del enfoque causal y la detección de rupturas.

## 🎯 Problema: Look-Ahead Bias

### ❌ Enfoque Tradicional (Con Sesgo)

```python
# Calcular PCA sobre TODA la serie histórica
X_all = df_features.values  # Incluye datos futuros
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)  # ← FIT usa TODO

pca = PCA(n_components=1)
pca.fit(X_scaled)  # ← FIT usa TODO

pc1 = pca.transform(X_scaled)
```

**Problema**: Para calcular `PC1[t]` (día t), el algoritmo **ve datos de días futuros** (t+1, t+2, ...).

Esto es **irreal** porque:
- Un trader real el día t no conoce el futuro
- Los parámetros de scaler/PCA están "contaminados" con información futura
- Los backtests resultan **optimistas** (rendimientos inflados)

---

### ✅ Enfoque Causal (Sin Sesgo)

```python
# Recorrer día por día
for t in range(min_inicio, n):
    # SOLO usar datos pasados [t-252:t]
    X_calib = X_all[t - ventana_entrenamiento : t]
    
    # Calibrar (fit) SOLO con pasado
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib)
    
    pca = PCA(n_components=1)
    pca.fit(X_calib_scaled)  # ← FIT SOLO pasado
    
    # Aplicar (transform, NO fit) a día actual
    x_t = X_all[t:t+1]
    x_t_scaled = scaler.transform(x_t)  # ← Transform, no fit
    pc1_t = pca.transform(x_t_scaled)
```

**Ventaja**: Simula exactamente lo que un trader hubiera visto en tiempo real.

---

## 📊 Pipeline de Análisis

### Paso 1: Generación de Características

```python
def generar_caracteristicas(precios):
    retornos = np.log(precios / precios.shift(1))
    
    for h in [5, 10, 21, 63]:
        ret_{h} = retornos.rolling(h).sum()      # Retorno acumulado
        vol_{h} = retornos.rolling(h).std()      # Volatilidad
```

**Resultado**: DataFrame con 8 características (4 retornos + 4 volatilidades)

---

### Paso 2: PCA Causal

**Cada `ventana_entrenamiento` días** (por defecto 252):

```
t = 252:  Entrenar PCA con [0:252]
          → PC1[252] = PCA.transform([252])
          
t = 253:  Entrenar PCA con [1:253]
          → PC1[253] = PCA.transform([253])
```

**Ventaja**: Sin look-ahead, cada PC1 usa solo datos históricos.

---

### Paso 3: Z-Score Causal

```python
# Estadísticos PRE-actuales (en ventana rodante de 252 días)
media_movil = PC1[t-252:t].mean()
desv_movil = PC1[t-252:t].std()

z_score[t] = (PC1[t] - media_movil) / desv_movil
```

---

### Paso 4: Filtrado de Eventos

Agrupa eventos cercanos (< `dias_exclusion` días) para evitar falsos positivos.

---

### Paso 5: Auditoría Pre/Post

Para cada evento auditado, calcula:
- **Shannon_pre/post**: Entropía información
- **Hugo_pre/post**: Entropía de fase (Hilbert)
- **Vol_pre/post**: Volatilidad
- **Ret_pre/post**: Retorno acumulado

---

## 🔬 Métricas de Entropía

### Entropía Shannon

```python
H = -Σ(p_i * log2(p_i))
```

- **H baja**: Retornos concentrados → **Orden**
- **H alta**: Retornos dispersos → **Caos**

### Entropía Hugo (Normalizada)

Basada en fase de Hilbert del rango [0, 2π]:

- **Hugo ≈ 0**: Fases concentradas → **Ciclos ordenados**
- **Hugo ≈ 1**: Fases uniformes → **Ciclos caóticos**

---

## 🎓 Caso de Estudio: Bitcoin Nov 2021

| Métrica | Valor | Interpretación |
|---|---|---|
| PC1 | +3.2 | Máximo histórico |
| Z-score | +2.89 | 99.5% confianza |
| Dirección | Positiva | Ruptura al alza |
| Delta_Hugo | +0.16 | Caos post-evento |
| Retorno_post | +0.134 | +13.4% en 21 días |

---

**Última actualización**: 2026-08-24
