"""
ts_analysis — пакет для анализа финансовых временных рядов
на тренд, mean reversion и режимные переключения.

Основные сводные функции:
    trend_diagnostics       — набор тестов на тренд
    mean_reversion_summary  — набор тестов на mean reversion
    regime_diagnostics      — режимная диагностика (VR, Hurst, ZA, CUSUM, MS-AR)
    johansen_test           — тест Йохансена на коинтеграцию
"""

from .trend import trend_diagnostics
from .mean_reversion import mean_reversion_summary
from .regime import regime_diagnostics
from .cointegration import johansen_test
