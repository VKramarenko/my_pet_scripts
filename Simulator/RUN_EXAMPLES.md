# Run Examples

Ниже собраны готовые команды запуска, чтобы быстро вспомнить рабочие сценарии.

## Где лежат примеры стратегий

После реорганизации примерные стратегии вынесены в отдельный пакет:

- `src/strategy/examples/buy_once.py`
- `src/strategy/examples/technical.py`
- `src/strategy/examples/__init__.py`

Базовый контракт стратегии остаётся отдельно в `src/strategy/base.py`.

## 1. Базовый запуск RSI-стратегии

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --strategy rsi_mean_reversion
```

## 2. RSI с явными параметрами

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --strategy rsi_mean_reversion --rsi-period 14 --oversold 30 --overbought 70 --qty 1
```

## 3. RSI с сохранением execution report

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --strategy rsi_mean_reversion --execution-report-json .\artifacts\execution_report.json --execution-orders-csv .\artifacts\execution_orders.csv --execution-fills-csv .\artifacts\execution_fills.csv
```

## 4. RSI с явной глубиной стакана

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --depth 5 --strategy rsi_mean_reversion
```

## 5. RSI с комиссиями и проскальзыванием

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --strategy rsi_mean_reversion --commission-model bps --commission-bps 5 --slippage-model fixed_bps --slippage-bps 2
```

## 6. RSI с лимитом на размер заявки

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --strategy rsi_mean_reversion --qty 2 --max-order-qty 2
```

## 7. Moving average стратегия

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --strategy moving_average_cross --short-window 3 --long-window 5 --qty 1
```

## 8. Passive strategy

```powershell
python .\run_simulation.py --csv ..\Modeller\test_data\test.csv --strategy passive_buy_once --price 100 --qty 1
```

## 9. Запуск дашборда по сохраненному execution report

```powershell
python .\view_execution_dash.py --csv ..\Modeller\test_data\test.csv --execution-report-json .\artifacts\execution_report.json
```

## 10. Дашборд с явной глубиной и нестандартным портом

```powershell
python .\view_execution_dash.py --csv ..\Modeller\test_data\test.csv --depth 5 --execution-report-json .\artifacts\execution_report.json --port 8060
```

## 11. Установка зависимостей для дашборда, если их нет

```powershell
python -m pip install dash plotly
```
