# Warren Buffett Jr 🎩📈

Sistema multi-agente de análisis de inversiones construido sobre **Claude Code** y la metodología **Ruta 2030 Wall Street Agent System v2.0.0**. Un agente principal (orquestador) coordina 6 sub-agentes especialistas que analizan una acción desde lentes independientes y producen un reporte final auditable de 100 puntos.

## Arquitectura

```
                    ┌──────────────────────┐
                    │   AGENTE PRINCIPAL   │
                    │  (orquestador — 100) │
                    └──────────┬───────────┘
        ┌──────────┬───────────┼───────────┬───────────┬──────────┐
        ▼          ▼           ▼           ▼           ▼          ▼
   Business   Financial     Market    Technical     Risk     Valuation
   Analysis   Analysis    & Growth   & Momentum  & Resilience Analysis
    20 pts     15 pts      20 pts      20 pts      15 pts     10 pts
```

Cada sub-agente corre **de forma independiente** — ninguno ve el score de otro hasta que los 6 outputs están congelados. Esto evita que la valuación contamine el análisis técnico, o que el momentum esconda fundamentales pobres.

## Regla innegociable

> **Sin evidencia, no hay número. Sin número, no hay score. Sin fórmula, no hay conclusión.**

Si falta data, el sistema responde `NOT_SCORABLE` — nunca reemplaza evidencia con confianza narrativa.

## Estructura del proyecto

| Carpeta | Contenido |
|---|---|
| `CLAUDE.md` | Instrucciones del orquestador (se cargan automáticamente en Claude Code) |
| `.claude/agents/` | Definiciones de los 6 sub-agentes |
| `Cerebro/` | Base de conocimiento completa: metodología, fórmulas, scoring, políticas de datos |
| `Perfil Inversionista/` | Perfil de riesgo y objetivos del inversionista |
| `Instrucciones/` | Documento original de instrucciones del agente |
| `API/` | Claves de API (privadas — no commitear nunca) |
| `Agente Principal/` | Workspace del orquestador |
| `Sub Agentes/` | Workspace y outputs de los especialistas |
| `Referencias/` | Material de referencia adicional |

## Inicio rápido (app web) 🚀

Requisitos: Python 3.11+ y git. No necesitas API keys para empezar — los
fundamentales vienen de SEC EDGAR (gratis) y el precio de Yahoo.

```bash
git clone https://github.com/infusionvictor/warren-buffett-jr.git
cd warren-buffett-jr/engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/webapp.py     # -> abre http://localhost:8765
```

En la página puedes:
- **Buscar** cualquier empresa de EE.UU. por ticker o nombre
- Ver la **explicación en palabras simples**, el **scorecard 1-10** de los
  agentes, la **gráfica con targets Bull/Medio/Bear** a 12 meses
- **✨ Descubrir empresas** — screener que encuentra compañías medianas,
  rentables y en crecimiento que quizás no conoces

También hay CLI: `wbj analyze NVDA`, `wbj scorecard NVDA`, `wbj screen`.

Opcional: copia tus API keys (FMP, FinnHub, FRED) en `API/.env` para
enriquecer los datos. Esa carpeta está en `.gitignore` — nunca se sube.

> ⚠️ Todo el output es **clasificación de research** con supuestos
> declarados — no es asesoría de inversión ni recomendación de compra/venta.

## Cómo usarlo (sistema multi-agente completo)

Desde esta carpeta, abre Claude Code y pide un análisis:

```
Analiza NVDA
```

El orquestador armará el packet de datos, correrá los 6 especialistas en paralelo, aplicará gates y overrides, y entregará el reporte final con niveles de confirmación/invalidación y trail de auditoría. *(El motor Python completo — indicadores técnicos, DCF institucional, los 6 especialistas — está en construcción: ver `RESUME.md`.)*

## Perfil del inversionista (resumen)

- **Objetivo:** crecimiento de capital, horizonte 3–5 años
- **Estilo:** agresivo y especulativo — acciones, ETF y opciones
- **Universo:** solo Estados Unidos, sin forex
- **Capital:** $25,000 USD — máx. 30–60% por posición
- **Prioridad:** probabilidad de éxito, timing de entradas y salidas

## Límites

Este sistema produce **research**, no órdenes. No promete retornos, no ejecuta trades y no convierte ningún nivel en una instrucción automática de compra/venta. Toda decisión final y ejecución es del inversionista.

---
*Metodología: Ruta 2030 Wall Street Agent System v2.0.0 — build 2026-07-14*
