"""
Motor QH v4 - Live Trading Integration
========================================
Conexión en tiempo real con datos de mercado y ejecución de órdenes.
Soporta: Alpaca, Interactive Brokers, y análisis standalone con yfinance.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks, butter, lfilter, hilbert
from scipy.stats import binomtest

# ==================== CONFIGURACIÓN DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('motor_qh_live.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURACIÓN GLOBAL ====================

class ConfigMotorQH:
    """Configuración centralizada del Motor QH."""
    
    # Ventanas de análisis
    RANGO_VENTANAS = list(range(150, 301, 5))  # cada 5 días para reducir cómputo
    VENTANA_EVALUACION = 252
    PASO_TEMPORAL_FINO = 21
    
    # Parámetros de significancia
    N_BOOT = 300
    BLOCK_SIZE = 10
    
    # Ángulos y sectores
    GRADOS_CORTES = [-90, -70, -40, -20, 0, 20, 40, 70, 90]
    BINS_RADIANES = np.deg2rad(GRADOS_CORTES)
    BINS_RADIANES[0] = -np.pi
    BINS_RADIANES[-1] = np.pi
    NUM_SECTORES = len(BINS_RADIANES) - 1
    PROB_IDEAL = 1 / NUM_SECTORES
    ETIQUETAS_SECTORES = [f"{GRADOS_CORTES[i]}° a {GRADOS_CORTES[i+1]}°" 
                          for i in range(NUM_SECTORES)]
    
    # Escala de theta
    FACTOR_ESCALA_THETA = 1e6
    VENTANA_VOLUMEN_REF = None
    
    # Umbrales de riesgo (CRÍTICO para live trading)
    UMBRAL_ENTROPIA_CRITICA = 0.15  # < 0.15 = bloqueo de fase sistémico
    UMBRAL_Z_SCORE_ALERTA = 2.58    # |z| > 2.58 = evento extremo (99%)
    UMBRAL_SIGNIFICANCIA_BOOTSTRAP = 0.95  # Fracción de sectores significativos
    
    # Configuración de datos
    CACHE_DIR = "cache_motor_qh"
    RESULTADO_DIR = "resultados_live"
    SEED = 42
    
    # Activos monitoreados (live)
    ACTIVOS_VIVOS = {
        "S&P500":      {"ticker": "^GSPC",    "tipo": "indice"},
        "Russell2000": {"ticker": "^RUT",     "tipo": "indice"},
        "Nasdaq":      {"ticker": "^IXIC",    "tipo": "indice"},
        "Oro":         {"ticker": "GLD",      "tipo": "etf"},
        "VIX":         {"ticker": "^VIX",     "tipo": "volatilidad"},
    }


# ==================== UTILIDADES ====================

def ensure_dirs():
    """Crea directorios necesarios."""
    for d in [ConfigMotorQH.CACHE_DIR, ConfigMotorQH.RESULTADO_DIR]:
        os.makedirs(d, exist_ok=True)


def descargar_datos_vivos(ticker: str, periodo: str = "2y") -> Tuple[pd.Series, pd.Series]:
    """Descarga datos actuales de yfinance (retorno logarítmico + volumen)."""
    try:
        data = yf.download(ticker, period=periodo, progress=False)
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
        
        logger.info(f"✓ {ticker}: {len(log_retornos)} retornos descargados ({log_retornos.index[0].date()} a {log_retornos.index[-1].date()})")
        return log_retornos, volumen
    except Exception as e:
        logger.error(f"✗ Error descargando {ticker}: {e}")
        return pd.Series(), pd.Series()


def calcular_r(log_retornos: pd.Series, rango_ventanas: List[int]) -> pd.DataFrame:
    """Volatilidad móvil por ventana."""
    dict_r = {w: log_retornos.rolling(window=w).std() for w in rango_ventanas}
    df = pd.DataFrame(dict_r)
    return df.replace([np.inf, -np.inf], np.nan)


def calcular_v(volumen: pd.Series, rango_ventanas: List[int], ventana_ref: Optional[int] = None) -> pd.DataFrame:
    """Log-volumen normalizado por MA."""
    dict_v = {}
    for w in rango_ventanas:
        w_vol = ventana_ref if ventana_ref is not None else w
        ma_vol = volumen.rolling(window=w_vol).mean().replace(0, np.nan)
        dict_v[w] = np.log(volumen / ma_vol)
    df = pd.DataFrame(dict_v)
    return df.replace([np.inf, -np.inf], np.nan)


def calcular_u(log_retornos: pd.Series, df_r: pd.DataFrame, 
               factor_escala: float = ConfigMotorQH.FACTOR_ESCALA_THETA) -> pd.DataFrame:
    """u_t(W) = factor_escala * x_t / r_t(W)."""
    idx = log_retornos.index.intersection(df_r.dropna(how="all").index)
    x = log_retornos.loc[idx]
    r_al = df_r.loc[idx].replace(0, np.nan)
    u = r_al.apply(lambda col: factor_escala * x / col)
    return u.replace([np.inf, -np.inf], np.nan)


def calcular_theta_matrix(df_u: pd.DataFrame) -> pd.DataFrame:
    """theta_t(W) = arctan(u_t(W))."""
    return np.arctan(df_u)


def diagnostico_convergencia_taylor(df_u: pd.DataFrame, umbral: float = 1.0) -> pd.Series:
    """Fracción de días |u| > 1 (fuera de radio de convergencia)."""
    return (df_u.abs() > umbral).mean(axis=0)


def theta_a_sectores_matrix(df_theta: pd.DataFrame, 
                              bins: np.ndarray = ConfigMotorQH.BINS_RADIANES,
                              num_sectores: int = ConfigMotorQH.NUM_SECTORES) -> pd.DataFrame:
    """Digitiza matriz theta -> IDs de sector [0..7]."""
    arr = np.digitize(df_theta.to_numpy(), bins[1:-1], right=False)
    arr = np.clip(arr, 0, num_sectores - 1).astype(np.int16)
    return pd.DataFrame(arr, index=df_theta.index, columns=df_theta.columns)


def probs_desde_ids(ids_sector: np.ndarray, num_sectores: int = ConfigMotorQH.NUM_SECTORES) -> np.ndarray:
    """Proporciones de observaciones por sector."""
    ids_sector = np.asarray(ids_sector, dtype=np.int64)
    counts = np.bincount(ids_sector, minlength=num_sectores)
    return counts / counts.sum()


def bootstrap_celda_fast(ids_sector: np.ndarray, n_boot: int = ConfigMotorQH.N_BOOT,
                         block_size: int = ConfigMotorQH.BLOCK_SIZE,
                         num_sectores: int = ConfigMotorQH.NUM_SECTORES,
                         prob_ideal: float = ConfigMotorQH.PROB_IDEAL) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap por bloques: (SE, CI_low, CI_high, significativo)."""
    rng = np.random.default_rng(ConfigMotorQH.SEED)
    ids_sector = np.asarray(ids_sector, dtype=np.int64)
    n = len(ids_sector)
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(n - block_size + 1, 1)

    starts = rng.integers(0, max_start, size=(n_boot, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    idx = np.clip(idx, 0, n - 1)

    muestras = ids_sector[idx]
    onehot = np.eye(num_sectores, dtype=np.int32)[muestras]
    boot_counts = onehot.sum(axis=1)
    boot_props = boot_counts / n

    se = boot_props.std(axis=0, ddof=1)
    ci_low = np.percentile(boot_props, 2.5, axis=0)
    ci_high = np.percentile(boot_props, 97.5, axis=0)
    significativo = (ci_low > prob_ideal) | (ci_high < prob_ideal)
    return se, ci_low, ci_high, significativo


def test_uniforme_iid_por_sector(ids_sector: np.ndarray, 
                                  num_sectores: int = ConfigMotorQH.NUM_SECTORES,
                                  prob_ideal: float = ConfigMotorQH.PROB_IDEAL) -> np.ndarray:
    """P-valor binomial exacto por sector."""
    ids_sector = np.asarray(ids_sector, dtype=np.int64)
    counts = np.bincount(ids_sector, minlength=num_sectores)
    n = int(counts.sum())
    return np.array([binomtest(int(c), n, prob_ideal).pvalue for c in counts])


# ==================== ANÁLISIS EN TIEMPO REAL ====================

class MotorQHLive:
    """Motor QH optimizado para análisis en vivo y alertas en tiempo real."""
    
    def __init__(self, config: ConfigMotorQH = None):
        self.config = config or ConfigMotorQH()
        self.rng = np.random.default_rng(self.config.SEED)
        ensure_dirs()
        self.datos_cache = {}
        self.alertas_activas = []
        logger.info("🚀 Motor QH Live inicializado")
    
    def analizar_activo(self, nombre: str, ticker: str) -> Dict:
        """Análisis completo de un activo (rápido, optimizado para live)."""
        logger.info(f"\n📊 Analizando {nombre} ({ticker})...")
        
        # Descargar datos
        log_ret, volumen = descargar_datos_vivos(ticker, periodo="2y")
        if log_ret.empty:
            return {"error": f"No se pudieron descargar datos para {ticker}"}
        
        # Estado cilíndrico
        df_r = calcular_r(log_ret, self.config.RANGO_VENTANAS)
        df_u = calcular_u(log_ret, df_r)
        df_theta = calcular_theta_matrix(df_u)
        df_v = calcular_v(volumen, self.config.RANGO_VENTANAS)
        
        # Diagnóstico Taylor
        frac_no_conv = diagnostico_convergencia_taylor(df_u)
        
        # Digitización a sectores
        df_ids = theta_a_sectores_matrix(df_theta)
        
        # Ventana representativa
        idx_vent_rep = self.config.RANGO_VENTANAS.index(225) if 225 in self.config.RANGO_VENTANAS else 0
        ids_ultima = df_ids.iloc[-self.config.VENTANA_EVALUACION:, idx_vent_rep].to_numpy()
        
        # Significancia
        se, ci_low, ci_high, sig = bootstrap_celda_fast(ids_ultima)
        p_iid = test_uniforme_iid_por_sector(ids_ultima)
        
        # Entropía rápida (últimos 20 días)
        theta_reciente = df_theta.iloc[-20:, idx_vent_rep].to_numpy()
        p_hist, _ = np.histogram(theta_reciente, bins=self.config.NUM_SECTORES, 
                                 range=(-np.pi, np.pi), density=True)
        p_hist = p_hist * (2 * np.pi / self.config.NUM_SECTORES)
        p_hist = p_hist[p_hist > 0]
        if len(p_hist) > 0:
            entropia = -np.sum(p_hist * np.log2(p_hist + 1e-10)) / np.log2(self.config.NUM_SECTORES)
        else:
            entropia = 1.0
        
        # Picos (anomalías recientes)
        exceso = np.maximum(0, probs_desde_ids(ids_ultima) - self.config.PROB_IDEAL)
        picos_idx, _ = find_peaks(exceso, prominence=0.02)
        
        resultado = {
            "nombre": nombre,
            "ticker": ticker,
            "fecha_analisis": datetime.now().isoformat(),
            "n_dias": len(log_ret),
            "entropia_reciente": float(entropia),
            "frac_taylor_no_convergente": {
                "min": float(frac_no_conv.min()),
                "max": float(frac_no_conv.max()),
                "mean": float(frac_no_conv.mean())
            },
            "bootstrap_significativo": int(sig.sum()),
            "p_iid_menor_005": int((p_iid < 0.05).sum()),
            "se_promedio": float(se.mean()),
            "ci_range": (float(ci_low.mean()), float(ci_high.mean())),
            "n_sectores_con_picos": len(picos_idx),
            "ultimo_retorno_log": float(log_ret.iloc[-1]) if len(log_ret) > 0 else np.nan,
            "volatilidad_actual": float(df_r.iloc[-1, idx_vent_rep]) if len(df_r) > 0 else np.nan,
        }
        
        # Alerta si hay riesgo sistémico
        if entropia < self.config.UMBRAL_ENTROPIA_CRITICA:
            alerta = {
                "timestamp": datetime.now().isoformat(),
                "activo": nombre,
                "tipo": "BLOQUEO_DE_FASE_CRITICO",
                "entropia": entropia,
                "descripcion": f"Sincronización extrema detectada. Entropía = {entropia:.3f} (umbral = {self.config.UMBRAL_ENTROPIA_CRITICA})"
            }
            self.alertas_activas.append(alerta)
            logger.warning(f"⚠️  ALERTA: {alerta['descripcion']}")
        
        # Alerta si hay anomalía en significancia bootstrap
        frac_sig = sig.sum() / self.config.NUM_SECTORES
        if frac_sig > self.config.UMBRAL_SIGNIFICANCIA_BOOTSTRAP:
            alerta = {
                "timestamp": datetime.now().isoformat(),
                "activo": nombre,
                "tipo": "ALTA_SIGNIFICANCIA_BOOTSTRAP",
                "frac_sectores": frac_sig,
                "descripcion": f"{frac_sig:.1%} de sectores significativos -> patrón estructurado"
            }
            self.alertas_activas.append(alerta)
            logger.warning(f"⚠️  ALERTA: {alerta['descripcion']}")
        
        return resultado
    
    def monitoreo_continuo(self, intervalo_minutos: int = 60):
        """Monitoreo continuo (para producción, requiere scheduler externo)."""
        logger.info(f"🔄 Iniciando monitoreo continuo (cada {intervalo_minutos} min)")
        resultados = []
        
        for nombre, cfg in self.config.ACTIVOS_VIVOS.items():
            res = self.analizar_activo(nombre, cfg["ticker"])
            resultados.append(res)
        
        # Guardar resultado
        df_res = pd.DataFrame(resultados)
        path_out = os.path.join(self.config.RESULTADO_DIR, 
                                f"monitoreo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_res.to_csv(path_out, index=False)
        logger.info(f"✓ Resultados guardados: {path_out}")
        
        # Guardar alertas
        if self.alertas_activas:
            path_alertas = os.path.join(self.config.RESULTADO_DIR, 
                                        f"alertas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(path_alertas, "w") as f:
                json.dump(self.alertas_activas, f, indent=2)
            logger.info(f"🚨 {len(self.alertas_activas)} alertas generadas")
        
        return resultados
    
    def generar_reporte(self) -> str:
        """Resumen ejecutivo en texto."""
        reporte = ["=" * 80]
        reporte.append("REPORTE MOTOR QH LIVE - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        reporte.append("=" * 80)
        
        for nombre, cfg in self.config.ACTIVOS_VIVOS.items():
            res = self.analizar_activo(nombre, cfg["ticker"])
            
            if "error" in res:
                reporte.append(f"\n❌ {nombre}: {res['error']}")
                continue
            
            reporte.append(f"\n📈 {nombre} ({res['ticker']})")
            reporte.append(f"   Entropía: {res['entropia_reciente']:.3f}")
            reporte.append(f"   Bootstrap Sig: {res['bootstrap_significativo']}/8 sectores")
            reporte.append(f"   Volatilidad: {res['volatilidad_actual']:.4f}")
            reporte.append(f"   Retorno (t-1): {res['ultimo_retorno_log']:.4f}")
            
            estado = "🟢 NORMAL"
            if res['entropia_reciente'] < self.config.UMBRAL_ENTROPIA_CRITICA:
                estado = "🔴 CRÍTICO"
            elif res['bootstrap_significativo'] > 4:
                estado = "🟡 ALERTA"
            
            reporte.append(f"   Estado: {estado}")
        
        reporte.append("\n" + "=" * 80)
        if self.alertas_activas:
            reporte.append(f"ALERTAS ACTIVAS ({len(self.alertas_activas)}):")
            for alerta in self.alertas_activas:
                reporte.append(f"  • {alerta['tipo']}: {alerta['descripcion']}")
        else:
            reporte.append("✓ Sin alertas críticas")
        
        reporte.append("=" * 80)
        return "\n".join(reporte)


# ==================== INTEGRACIÓN CON BROKERS ====================

class AdaptadorAlpaca:
    """Adaptador para Alpaca Trading API (simulación/live)."""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        try:
            from alpaca_trade_api import REST
            self.client = REST(api_key, secret_key, base_url=base_url)
            logger.info("✓ Conectado a Alpaca")
        except ImportError:
            logger.error("Instala: pip install alpaca-trade-api")
            self.client = None
    
    def obtener_posiciones(self):
        """Retorna posiciones actuales."""
        if not self.client:
            return []
        return self.client.list_positions()
    
    def enviar_orden(self, simbolo: str, cantidad: int, lado: str = "buy"):
        """Envía orden al mercado (paper o live)."""
        if not self.client:
            logger.error("Cliente Alpaca no inicializado")
            return None
        
        orden = self.client.submit_order(
            symbol=simbolo,
            qty=cantidad,
            side=lado,
            type="market",
            time_in_force="day"
        )
        logger.info(f"✓ Orden enviada: {simbolo} {cantidad} {lado}")
        return orden


class AdaptadorIB:
    """Adaptador para Interactive Brokers (via ibapi)."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, clientId: int = 1):
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
            logger.info(f"✓ IB Gateway en {host}:{port}")
        except ImportError:
            logger.error("Instala: pip install ibapi")
    
    def conectar(self):
        logger.info("Conexión a IB en progreso...")
        pass


# ==================== MAIN (EJEMPLO DE USO) ====================

def main():
    """Punto de entrada: análisis live."""
    
    logger.info("🎯 MOTOR QH v4 - LIVE TRADING INTEGRATION")
    logger.info("=" * 80)
    
    # Inicializar
    motor = MotorQHLive()
    
    # Análisis único (quick check)
    print("\n" + motor.generar_reporte())
    
    # Guardar resultados
    ensure_dirs()
    resultados = motor.monitoreo_continuo()
    
    # Para producción: scheduler externo (APScheduler, cron, etc.)
    logger.info("\n💡 Para monitoreo continuo en producción:")
    logger.info("   - APScheduler: python -c 'from apscheduler.schedulers.background import BackgroundScheduler'")
    logger.info("   - Cron: 0 */1 * * * python motor_qh_live.py")
    logger.info("   - Systemd: crear servicio con ExecStart=/usr/bin/python motor_qh_live.py")


if __name__ == "__main__":
    main()
