# Correr el motor `wbj` en Windows

El `README.md` de la raíz documenta rutas de macOS/Linux (`.venv/bin/…`). En
Windows el venv usa `.venv\Scripts\…`. Esta guía es el equivalente probado en
esta máquina (Python 3.12, PowerShell).

## Requisitos

- **Python 3.12** instalado (en esta máquina: `Python.Python.3.12` vía winget,
  en `%LOCALAPPDATA%\Programs\Python\Python312`).
- No necesitas claves de API para empezar: los fundamentales vienen de SEC
  EDGAR (gratis) y el precio de Yahoo Finance. Ver [`../API/README.md`](../API/README.md).

## Preparar el entorno (una sola vez)

Desde la carpeta `engine\`:

```powershell
# Crear el entorno virtual e instalar el paquete
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## App web (recomendado)

```powershell
.\webapp.ps1
```

Abre http://localhost:8765 — busca cualquier empresa de EE.UU. por ticker o
nombre, o toca **✨ Descubrir empresas** para el screener.

## CLI

Antes de usar la CLI en una sesión, fuerza UTF-8 (para acentos y símbolos):

```powershell
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
```

Luego:

```powershell
.\.venv\Scripts\wbj.exe analyze NVDA     # pipeline completo -> guarda reporte
.\.venv\Scripts\wbj.exe scorecard NVDA   # scorecard 1-10 por categoría
.\.venv\Scripts\wbj.exe screen           # descubrir empresas (research)
.\.venv\Scripts\wbj.exe track            # evaluar predicciones guardadas
```

Los reportes se guardan en `..\Reportes\<TICKER>\<YYYY-MM-DD>\`.

## Correr los tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ --ignore=tests\packet -q
```

> `tests\packet` se excluye porque `wbj\packet\builder.py` (Task 10 del plan del
> motor) aún no está implementado — ver [`../RESUME.md`](../RESUME.md). El resto
> del motor (180 tests) pasa.

## Las 6 categorías del scorecard

Todas las categorías ahora puntúan. Business / Financial / Risk salen de los
fundamentales de SEC EDGAR. Technical, Valuation y Market & Growth se calculan
desde precio/volumen (Yahoo), el benchmark SPY y los targets — ver
`wbj\specialists.py`, `wbj\indicators.py` y `wbj\marketdata.py`.

Regla de honestidad ("sin evidencia, no hay número"): una categoría con menos
de 70% de cobertura se muestra como **parcial** (con su puntaje y % de
cobertura) pero **no** cuenta en el overall, para no inflar la evidencia.
Resultado típico:

| Categoría | Estado | Por qué |
|---|---|---|
| Valuation | ✅ scored (~80%) | múltiplos PEG, yields vs tasa libre de riesgo, fair value, margen de seguridad |
| Technical & Momentum | ◑ parcial (~66%) | falta earnings-gap y patrones de ruptura (requieren fechas de earnings / registros de niveles) |
| Market & Growth | ◑ parcial (~35%) | falta TAM, revisiones de consenso y catalizadores (requieren keys FMP/FinnHub + investigación cualitativa del orquestador) |

Agregar las keys de API (ver [`../API/README.md`](../API/README.md)) permitiría
subir varias de estas dimensiones a cobertura completa en el futuro.

## Notas de Windows

- Todos los archivos de texto se escriben en **UTF-8 explícito** en el código
  (necesario porque el default de Windows es cp1252 y rompía con acentos y `≥`).
- Si `wbj` no aparece como comando, usa la ruta completa
  `.\.venv\Scripts\wbj.exe` o activa el venv con `.\.venv\Scripts\Activate.ps1`.
