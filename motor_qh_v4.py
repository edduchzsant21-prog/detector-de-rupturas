"""
Motor QH - Ocupación Angular Multi-Activo (v4)
================================================
v4 cambia el ORDEN de la definición de theta respecto a v3, a partir de
una revisión con desarrollo de Taylor:

  x_t     = ln(p_t / p_{t-1})                retorno logarítmico
  r_t(W)  = std_movil(x_t, ventana=W)        volatilidad local (= v3)
  z_t(W)  = x_t / r_t(W)                     retorno ESTANDARIZADO por ventana
  u_t(W)  = FACTOR_ESCALA_THETA * z_t(W)     argumento de arctan
  theta_t(W) = arctan(u_t(W))                YA NO se envuelve (ver nota)

CAMBIO DE ORDEN (v3 tenía FACTOR*arctan(x); v4 tiene arctan(FACTOR*x/r)):
  v3 multiplicaba el resultado de arctan por 10^6 -> sacaba theta del
  rango (-pi,pi] y había que "envolverlo" con angle(exp(i*theta)), lo que
  generaba muchas vueltas completas por variaciones mínimas en x_t (ver
  docstring de v3). v4 multiplica ANTES de entrar a arctan, así que el
  resultado queda naturalmente acotado a (-pi/2, pi/2) sin necesidad de
  envolver nada.

CAVEAT CRÍTICO DE ESCALA (por qué theta ahora SÍ depende de la ventana):
  Con FACTOR_ESCALA_THETA=1e6 y x_t sin estandarizar, arctan(1e6*x_t)
  SATURA casi siempre a +-pi/2 para cualquier x_t != 0 -> theta colapsa
  a signo(x_t), perdiendo toda la magnitud. Por eso v4 estandariza x_t
  por su propia volatilidad de ventana (z_t(W) = x_t/r_t(W), ~O(1)
  típicamente) ANTES de multiplicar por FACTOR_ESCALA_THETA. Esto es lo
  que reintroduce la dependencia de ventana en theta: cada W da una
  estimación de volatilidad distinta, y por lo tanto un z_t(W) distinto.

DESARROLLO DE TAYLOR (por qué importa el orden de magnitud de u):
  arctan(u) = u - u^3/3 + u^5/5 - ...  SOLO CONVERGE para |u| <= 1.
  Con FACTOR_ESCALA_THETA=1e6, u_t(W) es del orden de 1e5-1e6 casi
  siempre -> la serie de Taylor alrededor de u=0 NO CONVERGE, no es que
  sea "poco precisa". `diagnostico_convergencia_taylor` calcula, por
  cada ventana W, la fracción de días con |u_t(W)| > 1 (fuera del radio
  de convergencia) para que puedas calibrar FACTOR_ESCALA_THETA a un
  valor donde el desarrollo de Taylor tenga sentido matemático (orden 1
  a 5, aprox., si z_t(W) es O(1)).

SÁBANA DE PROBABILIDAD:
  Como theta(W) ahora sí varía por ventana, construir_tensor vuelve a
  producir prob/exceso/deficit/significancia con forma (tiempo, ventana,
  sector) -- una superficie completa para comparar ventanas entre sí,
  igual que el diseño original de v2, pero ahora con la definición
  correcta de theta y con r_medio/v_medio agregados por sector.

Se conserva de v3/v2:
1. bootstrap_celda_fast: robustez bajo remuestreo por bloques. NO es un
   p-valor contra H0 uniforme.
2. test_uniforme_iid_por_sector: p-valor formal (binomial exacto),
   asume iid -> anticonservador por autocorrelación real.
3. Sectores vía np.digitize.
4. picos_por_sector recorre los 8 sectores.

Caveats de modelización:
- CL=F: posibles artefactos de "roll" en los retornos.
- ^VIX: volumen y magnitud de retorno no comparables a equity/bonos/oro.
"""

import os
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks
from scipy.stats import binomtest

# ------------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------------

SEED = 42
RNG = np.random.default_rng(SEED)

HOY = pd.Timestamp.today().normalize()

ACTIVOS = {
    "S&P500":           {"ticker": "^GSPC",   "start": "2005-01-01"},
    "Oro (GLD)":         {"ticker": "GLD",     "start": "2005-01-01"},
    "Bonos LP (TLT)":    {"ticker": "TLT",     "start": "2005-01-01"},
    "Petroleo (CL=F)":   {"ticker": "CL=F",    "start": "2005-01-01"},
    "VIX":               {"ticker": "^VIX",    "start": "2005-01-01"},
    "BTC-USD":           {"ticker": "BTC-USD", "start": "2014-09-17"},
}

GRADOS_CORTES = [-90, -70, -40, -20, 0, 20, 40, 70, 90]
BINS_RADIANES = np.deg2rad(GRADOS_CORTES)
BINS_RADIANES[0] = -np.pi
BINS_RADIANES[-1] = np.pi
NUM_SECTORES = len(BINS_RADIANES) - 1
PROB_IDEAL = 1 / NUM_SECTORES
ETIQUETAS_SECTORES = [f"{GRADOS_CORTES[i]}° a {GRADOS_CORTES[i+1]}°" for i in range(NUM_SECTORES)]

RANGO_VENTANAS = list(range(150, 301))
VENTANA_EVALUACION = 252
PASO_TEMPORAL_FINO = 21
N_BOOT = 300
BLOCK_SIZE = 10

FACTOR_ESCALA_THETA = 1e6   # ver nota crítica en el docstring del módulo
VENTANA_VOLUMEN_REF = None  # None -> usa la misma ventana W que r_t para la MA de volumen

EVENTOS = {
    "Crisis 2008":    ("2007-06-01", "2009-06-01"),
    "Crash 2020":     ("2020-01-01", "2020-12-31"),
    "2024":            ("2024-01-01", "2024-12-31"),
    "Reciente (12m)": ((HOY - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), HOY.strftime("%Y-%m-%d")),
}

OUT_DIR = "resultados_motor_qh_fase_v4"
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 2. DESCARGA Y ÁNGULOS
# ------------------------------------------------------------------

def descargar_datos(ticker, start, end):
    """Descarga precio de cierre y volumen, y devuelve el retorno LOGARÍTMICO
    (no pct_change) porque theta_t se define sobre x_t = ln(p_t/p_t-1)."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"Sin datos para {ticker} entre {start} y {end}")

    if isinstance(data.columns, pd.MultiIndex):
        precios = data.xs("Close", axis=1, level=0).squeeze()
        volumen = data.xs("Volume", axis=1, level=0).squeeze()
    else:
        precios = data["Close"].squeeze()
        volumen = data["Volume"].squeeze()

    precios = precios.astype(float).dropna()
    volumen = volumen.astype(float).reindex(precios.index)

    log_retornos = np.log(precios / precios.shift(1))
    log_retornos = log_retornos.replace([np.inf, -np.inf], np.nan).dropna()
    volumen = volumen.reindex(log_retornos.index)
    return log_retornos, volumen


def calcular_u(log_retornos, df_r, factor_escala=FACTOR_ESCALA_THETA):
    """u_t(W) = factor_escala * x_t / r_t(W). Una columna por ventana.
    r_t(W) ES la volatilidad ya calculada por calcular_r -- se reutiliza,
    no se recalcula, para que theta y r queden acopladas por diseño."""
    idx = log_retornos.index.intersection(df_r.dropna(how="all").index)
    x = log_retornos.loc[idx]
    r_al = df_r.loc[idx].replace(0, np.nan)
    u = r_al.apply(lambda col: factor_escala * x / col)
    return u.replace([np.inf, -np.inf], np.nan)


def calcular_theta_matrix(df_u):
    """theta_t(W) = arctan(u_t(W)). Acotado naturalmente a (-pi/2, pi/2],
    no requiere wrap (a diferencia de v3)."""
    return np.arctan(df_u)


def taylor_arctan(u, orden=1):
    """Aproximación de Taylor de arctan(u) alrededor de u=0.
    Válida solo para |u| <= 1 (radio de convergencia de la serie)."""
    if orden == 1:
        return u
    elif orden == 3:
        return u - (u ** 3) / 3
    elif orden == 5:
        return u - (u ** 3) / 3 + (u ** 5) / 5
    else:
        raise ValueError("orden debe ser 1, 3 o 5")


def diagnostico_convergencia_taylor(df_u, umbral=1.0):
    """Por cada ventana W: fracción de días con |u_t(W)| > umbral, es decir
    FUERA del radio de convergencia de la serie de Taylor de arctan.
    Un valor alto (cercano a 1) significa que el desarrollo de Taylor no
    es aplicable en esa ventana con el FACTOR_ESCALA_THETA actual."""
    return (df_u.abs() > umbral).mean(axis=0)


def calcular_r(log_retornos, rango_ventanas=RANGO_VENTANAS):
    """r_t(W) = volatilidad móvil de x_t. Una columna por ventana."""
    dict_r = {w: log_retornos.rolling(window=w).std() for w in rango_ventanas}
    df = pd.DataFrame(dict_r)
    return df.replace([np.inf, -np.inf], np.nan)


def calcular_v(volumen, rango_ventanas=RANGO_VENTANAS, ventana_ref=VENTANA_VOLUMEN_REF):
    """v_t(W) = ln(Volumen_t / MA_W(Volumen_t)). Una columna por ventana.
    Si ventana_ref se fija, se usa esa única ventana para la MA de volumen
    en vez de acoplarla 1 a 1 con cada W de r_t."""
    dict_v = {}
    for w in rango_ventanas:
        w_vol = ventana_ref if ventana_ref is not None else w
        ma_vol = volumen.rolling(window=w_vol).mean().replace(0, np.nan)
        dict_v[w] = np.log(volumen / ma_vol)
    df = pd.DataFrame(dict_v)
    return df.replace([np.inf, -np.inf], np.nan)


def calcular_estado(log_retornos, volumen, rango_ventanas=RANGO_VENTANAS,
                     factor_escala_theta=FACTOR_ESCALA_THETA):
    """Arma el estado cilíndrico completo. theta, r y v son ahora los tres
    matriciales (tiempo x ventana), alineados al mismo índice de fechas."""
    df_r = calcular_r(log_retornos, rango_ventanas)
    df_u = calcular_u(log_retornos, df_r, factor_escala_theta)
    df_theta = calcular_theta_matrix(df_u)
    df_v = calcular_v(volumen, rango_ventanas)

    idx_comun = df_theta.dropna(how="all").index
    for df in (df_r, df_v):
        idx_comun = idx_comun.intersection(df.dropna(how="all").index)

    df_theta = df_theta.loc[idx_comun].dropna()
    df_r = df_r.loc[idx_comun].dropna()
    df_v = df_v.loc[idx_comun].dropna()
    df_u = df_u.loc[idx_comun].dropna()

    idx_final = df_theta.index.intersection(df_r.index).intersection(df_v.index)
    return df_theta.loc[idx_final], df_r.loc[idx_final], df_v.loc[idx_final], df_u.loc[idx_final]


def theta_a_sectores_matrix(df_theta, bins=BINS_RADIANES, num_sectores=NUM_SECTORES):
    """Digitiza la matriz completa (tiempo x ventana) -> IDs 0..7."""
    arr = np.digitize(df_theta.to_numpy(), bins[1:-1], right=False)
    arr = np.clip(arr, 0, num_sectores - 1).astype(np.int16)
    return pd.DataFrame(arr, index=df_theta.index, columns=df_theta.columns)


def probs_desde_ids(ids_sector, num_sectores=NUM_SECTORES):
    ids_sector = np.asarray(ids_sector, dtype=np.int64)
    counts = np.bincount(ids_sector, minlength=num_sectores)
    return counts / counts.sum()


# ------------------------------------------------------------------
# 3. SIGNIFICANCIA: DOS PRUEBAS DISTINTAS, NUNCA MEZCLADAS
# ------------------------------------------------------------------

def bootstrap_celda_fast(ids_sector, rng=RNG, n_boot=N_BOOT, block_size=BLOCK_SIZE,
                          num_sectores=NUM_SECTORES, prob_ideal=PROB_IDEAL):
    """
    Robustez de la proporción observada bajo remuestreo por bloques (preserva
    autocorrelación local). Devuelve SE, IC95% y un flag "significativo" que
    indica si el IC excluye la probabilidad uniforme. NO es un p-valor contra
    H0 uniforme -- es un chequeo de estabilidad del efecto bajo dependencia.
    """
    ids_sector = np.asarray(ids_sector, dtype=np.int64)
    n = len(ids_sector)
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(n - block_size + 1, 1)

    starts = rng.integers(0, max_start, size=(n_boot, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    idx = np.clip(idx, 0, n - 1)

    muestras = ids_sector[idx]                                   # (n_boot, n)
    onehot = np.eye(num_sectores, dtype=np.int32)[muestras]      # (n_boot, n, num_sectores)
    boot_counts = onehot.sum(axis=1)
    boot_props = boot_counts / n

    se = boot_props.std(axis=0, ddof=1)
    ci_low = np.percentile(boot_props, 2.5, axis=0)
    ci_high = np.percentile(boot_props, 97.5, axis=0)
    significativo = (ci_low > prob_ideal) | (ci_high < prob_ideal)
    return se, ci_low, ci_high, significativo


def test_uniforme_iid_por_sector(ids_sector, num_sectores=NUM_SECTORES, prob_ideal=PROB_IDEAL):
    """
    Test binomial exacto de dos colas, por sector, contra H0: p=1/num_sectores.
    P-VALOR FORMAL, pero asume observaciones iid. La autocorrelación real del
    ángulo lo vuelve previsiblemente ANTICONSERVADOR (p más chicos de lo real)
    -- usar como referencia, no como prueba definitiva por sí sola.
    """
    ids_sector = np.asarray(ids_sector, dtype=np.int64)
    counts = np.bincount(ids_sector, minlength=num_sectores)
    n = int(counts.sum())
    return np.array([binomtest(int(c), n, prob_ideal).pvalue for c in counts])


# ------------------------------------------------------------------
# 4. TENSOR WALK-FORWARD
# ------------------------------------------------------------------

def construir_tensor(df_theta, df_r, df_v, rango_ventanas=RANGO_VENTANAS,
                      ventana_evaluacion=VENTANA_EVALUACION,
                      paso_temporal=PASO_TEMPORAL_FINO, num_sectores=NUM_SECTORES):
    """
    Sabana de probabilidad: theta(W) vuelve a depender de la ventana (via la
    estandarizacion z_t(W)=x_t/r_t(W)), asi que prob/exceso/deficit/
    significancia/r_medio/v_medio salen con forma (tiempo, ventana, sector)
    -- comparable directamente entre ventanas 150..300.
    """
    df_ids = theta_a_sectores_matrix(df_theta)

    fechas_tensor = []
    prob_list, exceso_list, deficit_list = [], [], []
    sig_boot_list, p_iid_list = [], []
    r_medio_list, v_medio_list = [], []

    n_pasos = len(range(ventana_evaluacion, len(df_ids), paso_temporal))
    contador = 0
    for i in range(ventana_evaluacion, len(df_ids), paso_temporal):
        contador += 1
        corte_ids = df_ids.iloc[i - ventana_evaluacion:i]
        corte_r = df_r.iloc[i - ventana_evaluacion:i]
        corte_v = df_v.iloc[i - ventana_evaluacion:i]
        fecha_corte = corte_ids.index[-1]
        fechas_tensor.append(fecha_corte)

        prob_t, exc_t, def_t, sig_t, piid_t, r_t, v_t = [], [], [], [], [], [], []
        for ventana in rango_ventanas:
            ids_ventana = corte_ids[ventana].to_numpy()

            prob_emp = probs_desde_ids(ids_ventana, num_sectores)
            diff = prob_emp - PROB_IDEAL
            exc_t.append(np.maximum(0, diff))
            def_t.append(np.maximum(0, -diff))
            prob_t.append(prob_emp)

            _, _, _, sig = bootstrap_celda_fast(ids_ventana, num_sectores=num_sectores)
            sig_t.append(sig)

            p_iid = test_uniforme_iid_por_sector(ids_ventana, num_sectores=num_sectores)
            piid_t.append(p_iid)

            r_corte = corte_r[ventana].to_numpy()
            v_corte = corte_v[ventana].to_numpy()
            r_medio_sector = np.full(num_sectores, np.nan)
            v_medio_sector = np.full(num_sectores, np.nan)
            for s in range(num_sectores):
                mask = ids_ventana == s
                if mask.any():
                    r_medio_sector[s] = np.nanmean(r_corte[mask])
                    v_medio_sector[s] = np.nanmean(v_corte[mask])
            r_t.append(r_medio_sector)
            v_t.append(v_medio_sector)

        prob_list.append(np.array(prob_t))
        exceso_list.append(np.array(exc_t))
        deficit_list.append(np.array(def_t))
        sig_boot_list.append(np.array(sig_t))
        p_iid_list.append(np.array(piid_t))
        r_medio_list.append(np.array(r_t))
        v_medio_list.append(np.array(v_t))

        if contador % 5 == 0 or contador == n_pasos:
            print(f"    corte {contador}/{n_pasos} ({fecha_corte.date()})")

    return {
        "fechas": fechas_tensor,
        "prob": np.stack(prob_list, axis=0),
        "exceso": np.stack(exceso_list, axis=0),
        "deficit": np.stack(deficit_list, axis=0),
        "significativo_bootstrap": np.stack(sig_boot_list, axis=0),
        "p_valor_iid": np.stack(p_iid_list, axis=0),
        "r_medio": np.stack(r_medio_list, axis=0),
        "v_medio": np.stack(v_medio_list, axis=0),
    }



# ------------------------------------------------------------------
# 5. PICOS -> FECHAS REALES (LOS 8 SECTORES)
# ------------------------------------------------------------------

def picos_todos_los_sectores(resultado, ventana_representativa=225, prominence=0.02):
    idx_ventana = RANGO_VENTANAS.index(ventana_representativa)
    filas = []
    for s_idx, etiqueta in enumerate(ETIQUETAS_SECTORES):
        serie = resultado["exceso"][:, idx_ventana, s_idx]  # (tiempo,) -- por ventana representativa
        picos_idx, _ = find_peaks(serie, prominence=prominence)
        for i in picos_idx:
            filas.append({
                "sector": etiqueta,
                "fecha": resultado["fechas"][i],
                "exceso": serie[i],
                "significativo_bootstrap": bool(resultado["significativo_bootstrap"][i, idx_ventana, s_idx]),
                "p_valor_iid": resultado["p_valor_iid"][i, idx_ventana, s_idx],
                "r_medio": resultado["r_medio"][i, idx_ventana, s_idx],
                "v_medio": resultado["v_medio"][i, idx_ventana, s_idx],
            })
    return pd.DataFrame(filas).sort_values(["sector", "exceso"], ascending=[True, False])


# ------------------------------------------------------------------
# 6. RESUMEN POR EVENTO DE MERCADO
# ------------------------------------------------------------------

def resumen_por_evento(resultado, nombre_activo, ventana_representativa=225):
    idx_ventana = RANGO_VENTANAS.index(ventana_representativa)
    filas = []
    fechas = pd.DatetimeIndex(resultado["fechas"])
    for nombre_evento, (ini, fin) in EVENTOS.items():
        mask = (fechas >= ini) & (fechas <= fin)
        if mask.sum() == 0:
            continue
        exceso_medio = resultado["exceso"][mask, idx_ventana, :].mean(axis=0)
        deficit_medio = resultado["deficit"][mask, idx_ventana, :].mean(axis=0)
        frac_sig_boot = resultado["significativo_bootstrap"][mask, idx_ventana, :].mean(axis=0)
        frac_p_iid_05 = (resultado["p_valor_iid"][mask, idx_ventana, :] < 0.05).mean(axis=0)
        r_medio_evento = np.nanmean(resultado["r_medio"][mask, idx_ventana, :], axis=0)
        v_medio_evento = np.nanmean(resultado["v_medio"][mask, idx_ventana, :], axis=0)
        for s_idx, etiqueta in enumerate(ETIQUETAS_SECTORES):
            filas.append({
                "activo": nombre_activo,
                "evento": nombre_evento,
                "sector": etiqueta,
                "exceso_medio": exceso_medio[s_idx],
                "deficit_medio": deficit_medio[s_idx],
                "frac_significativo_bootstrap": frac_sig_boot[s_idx],
                "frac_p_iid_menor_0.05": frac_p_iid_05[s_idx],
                "r_medio": r_medio_evento[s_idx],
                "v_medio": v_medio_evento[s_idx],
            })
    return pd.DataFrame(filas)


# ------------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------------

def main():
    resumen_global = []

    for nombre_activo, cfg in ACTIVOS.items():
        print(f"\n=== {nombre_activo} ({cfg['ticker']}) ===")
        if cfg["ticker"] == "CL=F":
            print("  [aviso] futuro continuo: posibles artefactos de roll en los retornos")
        if cfg["ticker"] == "^VIX":
            print("  [aviso] retorno % de VIX no es económicamente comparable a equity/bonos/oro")

        try:
            log_retornos, volumen = descargar_datos(cfg["ticker"], cfg["start"], HOY.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"  ERROR descargando {nombre_activo}: {e}")
            continue

        print(f"  {len(log_retornos)} retornos diarios, {log_retornos.index[0].date()} a {log_retornos.index[-1].date()}")
        df_theta, df_r, df_v, df_u = calcular_estado(log_retornos, volumen)
        print(f"  Estado (theta, r, v) calculado: {len(df_theta)} fechas x {df_r.shape[1]} ventanas")

        frac_no_convergente = diagnostico_convergencia_taylor(df_u)
        print(f"  Taylor: fracción de días |u|>1 (no converge) -> "
              f"min={frac_no_convergente.min():.2f}, max={frac_no_convergente.max():.2f} "
              f"a través de las ventanas 150-300")

        resultado = construir_tensor(df_theta, df_r, df_v)

        path_pkl = os.path.join(OUT_DIR, f"tensor_{nombre_activo.replace(' ', '_')}.pkl")
        with open(path_pkl, "wb") as f:
            pickle.dump(resultado, f)
        print(f"  Guardado: {path_pkl}")

        picos = picos_todos_los_sectores(resultado)
        picos.to_csv(os.path.join(OUT_DIR, f"picos_{nombre_activo.replace(' ', '_')}_todos_sectores.csv"), index=False)

        resumen_activo = resumen_por_evento(resultado, nombre_activo)
        resumen_global.append(resumen_activo)

    if resumen_global:
        tabla_final = pd.concat(resumen_global, ignore_index=True)
        tabla_final.to_csv(os.path.join(OUT_DIR, "resumen_eventos_todos_los_activos.csv"), index=False)
        print("\n=== RESUMEN FINAL (por activo x evento x sector) ===")
        print(tabla_final.to_string(index=False))


if __name__ == "__main__":
    main()
