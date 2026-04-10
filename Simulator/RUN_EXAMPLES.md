# Run Examples

Ниже собраны готовые команды запуска, чтобы быстро вспомнить рабочие сценарии.

## Где лежат примеры стратегий

Каждая стратегия живёт в своём файле внутри пакета `src/strategy/examples/`:

| Файл | Класс | Описание |
|---|---|---|
| `buy_once.py` | `PassiveBuyOnceStrategy` | Одна пассивная заявка на покупку |
| `rsi_mean_reversion.py` | `RSIMeanReversionStrategy` | RSI market-order mean reversion |
| `rsi_limit_order_timeout.py` | `RSILimitOrderTimeoutStrategy` | RSI пассивные лимиты + отмена по TTL |
| `rsi_limit_order_template.py` | `RSILimitOrderTemplateStrategy` | Обучающий шаблон RSI lifecycle |
| `rsi_dual_book_timeout.py` | `RSIDualBookTimeoutStrategy` | RSI в нескольких стаканах независимо + TTL |
| `moving_average_cross.py` | `MovingAverageCrossStrategy` | Moving average crossover |
| `indicators.py` | `compute_rsi` | Общие индикаторы |

Базовый контракт стратегии — `src/strategy/base.py`.

Добавить новую стратегию: создать файл в `src/strategy/examples/`, зарегистрировать в `src/strategy/runtime.py`.

---

## Режимы запуска `run_simulation.py`

### Один стакан (`--csv`)

```powershell
python .\run_simulation.py --csv <path_to_csv> --strategy <name> [options]
```

`instrument_id` читается автоматически из колонки `symbol` CSV.

### Несколько стаканов (`--trading-csv` / `--info-csv`)

```powershell
python .\run_simulation.py `
  --trading-csv <path1> `
  --trading-csv <path2> `
  --info-csv    <path3> `
  --strategy rsi_dual_book_timeout [options]
```

- `--trading-csv` (повторяемый) — стаканы, в которых разрешено торговать
- `--info-csv` (повторяемый) — стаканы только для чтения (ордера отклоняются с предупреждением)
- `instrument_id` для каждого файла берётся из колонки `symbol` первой строки

---

## Тестовые данные

Файлы в `test_data/`:

| Файл | symbol | Роль |
|---|---|---|
| `ob_big.csv` | `BTCUSDT` | основной торговый стакан |
| `ob_big_shifted.csv` | `ALTUSDT` | второй торговый стакан (сдвинутые цены, +6 с) |
| `ob_big_shifted_info.csv` | `INFUSDT` | информационный стакан (цены ×5, не торгуем) |

Чтобы сгенерировать shifted/info файлы из `ob_big.csv`:

```powershell
python .\generate_shifted_orderbook.py
```

Конфигурация генерации — в блоке `CONFIG` внутри скрипта.

---

## 1. Один стакан — RSI limit order с timeout

```powershell
python .\run_simulation.py `
  --csv .\test_data\ob_big.csv `
  --strategy rsi_limit_order_timeout `
  --rsi-period 14 --oversold 30 --overbought 70 `
  --qty 1 --order-ttl-seconds 5
```

## 2. Один стакан — RSI limit order с timeout + сохранение отчёта

```powershell
python .\run_simulation.py `
  --csv .\test_data\ob_big.csv `
  --strategy rsi_limit_order_timeout `
  --rsi-period 14 --oversold 30 --overbought 70 `
  --qty 1 --order-ttl-seconds 5 `
  --execution-report-json .\artifacts\execution_report.json `
  --execution-orders-csv  .\artifacts\execution_orders.csv `
  --execution-fills-csv   .\artifacts\execution_fills.csv
```

## 3. Один стакан — визуализация дашборда

Запускается после того как сохранён execution report (см. пример 2):

```powershell
python .\view_execution_dash.py `
  --csv .\test_data\ob_big.csv `
  --execution-report-json .\artifacts\execution_report.json
```

Открыть в браузере: **http://127.0.0.1:8050**

Дополнительные параметры:

```powershell
python .\view_execution_dash.py `
  --csv .\test_data\ob_big.csv `
  --execution-report-json .\artifacts\execution_report.json `
  --port 8051 --debug
```

---

## 4. Два торговых стакана + информационный — RSI dual book

Полный пайплайн:

### Шаг 1 — запуск бэктеста

```powershell
python .\run_simulation.py `
  --trading-csv .\test_data\ob_big.csv `
  --trading-csv .\test_data\ob_big_shifted.csv `
  --info-csv    .\test_data\ob_big_shifted_info.csv `
  --strategy rsi_dual_book_timeout `
  --rsi-period 14 --oversold 30 --overbought 70 `
  --qty 1 --order-ttl-seconds 5 `
  --execution-report-json .\artifacts\execution_report_dual.json `
  --execution-orders-csv  .\artifacts\execution_orders_dual.csv `
  --execution-fills-csv   .\artifacts\execution_fills_dual.csv
```

Стратегия торгует в `BTCUSDT` и `ALTUSDT` независимо (RSI по каждому),
`INFUSDT` получает тики через `on_snapshot` но ордера на него отклоняются.
Репорт содержит колонку `instrument_id` в orders и fills.

### Шаг 2 — дашборд с раздельными графиками по инструментам

Передаём оба торговых CSV — каждый получит отдельный двухпанельный график
(цены + позиция), в котором отображаются только сделки своего инструмента:

```powershell
python .\view_execution_dash.py `
  --trading-csv .\test_data\ob_big.csv `
  --trading-csv .\test_data\ob_big_shifted.csv `
  --execution-report-json .\artifacts\execution_report_dual.json
```

Открыть в браузере: **http://127.0.0.1:8050**

На странице будет:
- График **BTCUSDT** — цены + ордера BTCUSDT + позиция BTCUSDT
- График **ALTUSDT** — цены + ордера ALTUSDT + позиция ALTUSDT
- Общая таблица всех ордеров (с колонкой `instrument_id`)
- Общая таблица всех fills (с колонкой `instrument_id`)

---

## 5. Один стакан — Moving average crossover

```powershell
python .\run_simulation.py `
  --csv .\test_data\ob_big.csv `
  --strategy moving_average_cross `
  --short-window 3 --long-window 5 --qty 1
```

## 6. Один стакан — RSI mean reversion с комиссиями

```powershell
python .\run_simulation.py `
  --csv .\test_data\ob_big.csv `
  --strategy rsi_mean_reversion `
  --rsi-period 14 --oversold 30 --overbought 70 --qty 1 `
  --commission-model bps --commission-bps 5 `
  --slippage-model fixed_bps --slippage-bps 2
```

## 7. Один стакан — RSI template

```powershell
python .\run_simulation.py `
  --csv .\test_data\ob_big.csv `
  --strategy rsi_limit_order_template `
  --rsi-period 14 --oversold 30 --overbought 70 `
  --qty 1 --order-ttl-seconds 5
```

## 8. Один стакан — passive buy once

```powershell
python .\run_simulation.py `
  --csv .\test_data\ob_big.csv `
  --strategy passive_buy_once --price 65000 --qty 1
```

---

## Установка зависимостей дашборда

```powershell
python -m pip install dash plotly
```

## Запуск тестов

```powershell
python -m pytest tests/ -v
```
