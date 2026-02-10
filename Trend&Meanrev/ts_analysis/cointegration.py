"""
Модуль теста Йохансена на коинтеграцию.

Функции:
    johansen_test — запускает тест Йохансена и возвращает dict с результатами

Класс:
    JohansenCointegrationTest — полная реализация с деталями
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Union, Any

from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")


def johansen_test(data: Union[pd.DataFrame, np.ndarray, dict],
                  column_names: Optional[List[str]] = None,
                  det_order: int = -1,
                  k_ar_diff: int = 1,
                  alpha: float = 0.05) -> Dict[str, Any]:
    """
    Тест Йохансена на коинтеграцию нескольких временных рядов.

    Параметры:
        data         — DataFrame, 2D ndarray или dict с рядами по столбцам
        column_names — названия рядов (если data — ndarray)
        det_order    — детерминистический член: -1 (нет), 0 (drift), 1 (drift+trend)
        k_ar_diff    — лаги разностей в VAR-модели
        alpha        — уровень значимости

    Возвращает dict с ключами:
        test_parameters, eigenvalues, hypotheses,
        cointegration_vectors, conclusion
    """
    jt = JohansenCointegrationTest(alpha=alpha)
    jt.run_test(data, column_names=column_names,
                det_order=det_order, k_ar_diff=k_ar_diff)
    return jt.results


class JohansenCointegrationTest:
    """Реализация теста Йохансена на коинтеграцию."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results: Dict[str, Any] = {}

    def _prepare_data(self, data, column_names):
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if column_names is None:
                column_names = [f"Series_{i+1}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=column_names)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Поддерживаются DataFrame, numpy array или dict.")

        df = df.dropna()
        if df.shape[0] < 50:
            warnings.warn(f"Мало наблюдений: {df.shape[0]}. Рекомендуется 100+.")
        if df.shape[1] < 2:
            raise ValueError("Для теста Йохансена нужно минимум 2 ряда.")
        return df

    def run_test(self, data, column_names=None,
                 det_order: int = -1, k_ar_diff: int = 1,
                 significance_level: str = "5%") -> Dict[str, Any]:
        """Запуск теста Йохансена."""
        df = self._prepare_data(data, column_names)
        n_series = df.shape[1]
        series_names = list(df.columns)

        result = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)

        crit_idx = {"1%": 0, "5%": 1, "10%": 2}.get(significance_level, 1)

        # Параметры теста
        self.results["test_parameters"] = {
            "n_series": n_series,
            "n_observations": df.shape[0],
            "series_names": series_names,
            "det_order": det_order,
            "k_ar_diff": k_ar_diff,
            "significance_level": significance_level,
        }

        # Собственные значения
        self.results["eigenvalues"] = sorted(result.eig.tolist(), reverse=True)

        # Гипотезы
        hypotheses = []
        for i in range(n_series):
            trace_stat = result.lr1[i]
            trace_crit = result.cvt[i][crit_idx]
            max_stat = result.lr2[i]
            max_crit = result.cvm[i][crit_idx]

            hypotheses.append({
                "r": i,
                "trace_statistic": float(trace_stat),
                "trace_critical_value": float(trace_crit),
                "trace_reject_h0": bool(trace_stat > trace_crit),
                "max_eigen_statistic": float(max_stat),
                "max_eigen_critical_value": float(max_crit),
                "max_eigen_reject_h0": bool(max_stat > max_crit),
            })
        self.results["hypotheses"] = hypotheses

        # Коинтеграционные векторы (нормализованные)
        vectors = result.evec.T
        normalized = []
        for vec in vectors:
            for val in vec:
                if abs(val) > 1e-10:
                    normalized.append([float(v / val) for v in vec])
                    break
            else:
                normalized.append([float(v) for v in vec])
        self.results["cointegration_vectors"] = normalized

        # Итоговое заключение
        r_trace = sum(1 for h in hypotheses if h["trace_reject_h0"])
        r_max = sum(1 for h in hypotheses if h["max_eigen_reject_h0"])
        recommended = min(r_trace, r_max)

        if recommended == 0:
            assessment = "No cointegration"
            recommendation = "Ряды не коинтегрированы. Mean reversion стратегия рискованна."
        elif recommended < n_series:
            assessment = f"{recommended} cointegration relation(s)"
            recommendation = "Ряды коинтегрированы. Подходит для mean reversion / pair trading."
        else:
            assessment = "Full cointegration"
            recommendation = "Полная коинтеграция. Нужна осторожность."

        self.results["conclusion"] = {
            "cointegration_vectors_trace": r_trace,
            "cointegration_vectors_max_eigen": r_max,
            "recommended": recommended,
            "is_cointegrated": recommended > 0,
            "assessment": assessment,
            "recommendation": recommendation,
        }

        return self.results
