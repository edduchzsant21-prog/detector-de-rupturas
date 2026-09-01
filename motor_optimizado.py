import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt
from numba import njit

# =====================================================================
# CONFIGURACIÓN NUMBA: PC1 RODANTE (SIN SVD)
# =====================================================================
@njit
def _power_pc1_rodante_numba(X_vals, ventana, n_iter=25):
    """
    Calcula PC1 del último elemento de cada ventana móvil:
      - z-score por columna dentro de la ventana (sin look-ahead)
      - covarianza C = (1/ventana) X^T X
      - PC1 via power iteration (autovector del mayor autovalor)
    """
    n_filas, n_cols = X_vals.shape
    pc1_vector = np.full(n_filas, np.nan)

    # vector inicial
    v = np.empty(n_cols, dtype=np.float64)
    for k in range(n_cols):
        v[k] = 1.0 / n_cols

    Xw = np.empty((ventana, n_cols), dtype=np.float64)
    C = np.empty((n_cols, n_cols), dtype=np.float64)

    eps = 1e-12

    for i in range(ventana - 1, n_filas):
        # --- 1) Copiar ventana histórica estricta + z-score manual por columna
        for r in range(ventana):
            base_row = i - ventana + 1 + r
            for c in range(n_cols):
                Xw[r, c] = X_vals[base_row, c]

        for c in range(n_cols):
            # media
            mu = 0.0
            for r in range(ventana):
                mu += Xw[r, c]
            mu /= ventana

            # var
            var = 0.0
            for r in range(ventana):
                d = Xw[r, c] - mu
                var += d * d
            var /= ventana

            sd = np.sqrt(var)
            if sd > 1e-8:
                for r in range(ventana):
                    Xw[r, c] = (Xw[r, c] - mu) / sd
            else:
                for r in range(ventana):
                    Xw[r, c] = (Xw[r, c] - mu)

        # --- 2) Covarianza: C = (1/ventana) Xw^T Xw
        for a in range(n_cols):
            for b in range(n_cols):
                s = 0.0
                for r in range(ventana):
                    s += Xw[r, a] * Xw[r, b]
                C[a, b] = s / ventana

        # --- 3) Power iteration para el mayor autovector
        for it in range(n_iter):
            # w = C v
            w = np.empty(n_cols, dtype=np.float64)
            for j in range(n_cols):
                tmp = 0.0
                for l in range(n_cols):
                    tmp += C[j, l] * v[l]
                w[j] = tmp

            # normalizar
            norm = 0.0
            for j in range(n_cols):
                norm += w[j] * w[j]
            norm = np.sqrt(norm) + eps

            for j in range(n_cols):
                v[j] = w[j] / norm

        # --- 4) PC1 del TICK ACTUAL (última fila z-scoreada): x_last · v
        last = 0.0
        for c in range(n_cols):
            last += Xw[ventana - 1, c] * v[c]
        pc1_vector[i] = last

    return pc1_vector


# =====================================================================
# MODULOS OPTIMIZADOS (SIN LOOK-AHEAD BIAS Y CON CORRECCIONES)
# =====================================================================
class MotorAuditoriaRupturas:
    """
    Motor cuántico de auditoría multiescala optimizado con Numba.
    """
    def __init__(self, horizontes=[5, 10, 21, 63], ventana_zscore=252):
        self.horizontes = horizontes
        self.ventana_zscore = ventana_zscore

    def generar_caracteristicas(self, precios: pd.Series) -> pd.DataFrame:
        """Genera matriz de características multiescala."""
        df_features = pd.DataFrame(index=precios.index)
        retornos = np.log(precios / precios.shift(1))

        for h in self.horizontes:
            df_features[f'ret_{h}'] = retornos.rolling(window=h).sum()
            df_features[f'vol_{h}'] = retornos.rolling(window=h).std()

        return df_features.dropna()

    def detectar_rupturas(self, df_features: pd.DataFrame, precios: pd.Series) -> pd.DataFrame:
        """
        Calcula PC1 y Z-Score con PCA rodante en Numba (sin SVD) y z-score robusto.
        """
        X_vals = df_features.values

        # PCA rodante (power iteration)
        pc1_rodante = _power_pc1_rodante_numba(X_vals, self.ventana_zscore, n_iter=25)

        df_resultados = pd.DataFrame(index=df_features.index)
        df_resultados['PC1'] = pc1_rodante
        df_resultados['Precio'] = precios.loc[df_resultados.index]

        # Z-Score móvil (protegido)
        media_movil = df_resultados['PC1'].rolling(window=self.ventana_zscore).mean()
        std_movil = df_resultados['PC1'].rolling(window=self.ventana_zscore).std()
        std_movil = std_movil.clip(lower=1e-8)  # evita división por ~0

        df_resultados['Z_score'] = (df_resultados['PC1'] - media_movil) / std_movil

        return df_resultados.dropna()

    def calcular_modelo_ls(self, serie_retornos: pd.Series, n_bins: int = 12) -> float:
        """
        Entropía sobre fase instantánea tras filtro Butterworth estable.
        Corrección: distribución prob. normalizada consistentemente.
        """
        x = serie_retornos.dropna().values
        if len(x) < 30:
            return 0.0

        # 1) Centrado
        x_centered = x - np.mean(x)

        # 2) Filtro pasabanda (estable)
        try:
            b, a = butter(2, [0.05, 0.85], btype='band')
            x_filtrado = filtfilt(b, a, x_centered)
        except:
            x_filtrado = x_centered

        # 3) Hilbert y fase
        analitica = hilbert(x_filtrado)
        fase_instantanea = np.angle(analitica)
        theta = np.mod(fase_instantanea, 2 * np.pi)

        # 4) Histograma -> probabilidades normalizadas
        p, _ = np.histogram(theta, bins=n_bins, range=(0, 2 * np.pi), density=False)
        total = p.sum()
        if total <= 0:
            return 0.0
        p = p / total

        # 5) Quitar ceros sin romper normalización
        p = p[p > 0]
        p = p / p.sum()

        # 6) Entropía normalizada
        S_H = -np.sum(p * np.log2(p))
        S_H_norm = S_H / np.log2(n_bins) if n_bins > 1 else S_H
        return float(S_H_norm)


def detectar_puntos_inflexion_qh(serie_precios: pd.Series, multiplicador_vol: float = 0.15, ventana_vol: int = 21) -> pd.DataFrame:
    """
    Tolerancia Adaptativa para QH.
    """
    df = pd.DataFrame({'precio': serie_precios})

    df['retorno_log'] = np.log(df['precio'] / df['precio'].shift(1))
    df['theta'] = np.arctan(df['retorno_log'])

    # Segunda diferencia del frente angular
    df['qh'] = df['theta'] - 2 * df['theta'].shift(1) + df['theta'].shift(2)

    # Umbral adaptativo móvil
    df['tolerancia_dinamica'] = df['qh'].rolling(window=ventana_vol).std() * multiplicador_vol

    # Inflexión
    df['es_cero'] = np.abs(df['qh']) <= df['tolerancia_dinamica']

    return df.dropna()


# =====================================================================
# PIPELINE DE PRUEBA INTEGRAL
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    pasos = 600
    precios_simulados = pd.Series(
        100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, size=pasos)))
    )

    print("--- 1. Ejecutando Inflexión QH con Tolerancia Adaptativa ---")
    df_inflexion = detectar_puntos_inflexion_qh(precios_simulados, multiplicador_vol=0.20)
    puntos_calma = df_inflexion[df_inflexion['es_cero']]
    print(f"Puntos de compresión/inflexión detectados dinámicamente: {len(puntos_calma)}")

    print("\n--- 2. Ejecutando Auditoría con PCA Rodante Numba (Sin SVD) ---")
    auditor = MotorAuditoriaRupturas(horizontes=[3, 5, 10, 21], ventana_zscore=100)

    features = auditor.generar_caracteristicas(precios_simulados)
    rupturas = auditor.detectar_rupturas(features, precios_simulados)

    print(f"Registros procesados de forma instantánea: {len(rupturas)}")
    print(rupturas[['Precio', 'PC1', 'Z_score']].tail(3))

    print("\n--- 3. Evaluando Caos Espectral Estable (Modelo LS) ---")
    retornos = np.log(precios_simulados / precios_simulados.shift(1))
    entropia_ls = auditor.calcular_modelo_ls(retornos)
    print(f"Entropía de Hugo Estabilizada (Filtro Butterworth): {entropia_ls:.4f}")
