#!/usr/bin/env python3
"""Pontua as previsões 1X2 da Copa 2026 contra os resultados reais.

Lê a submissão (data/results/predictions_submission.csv), o calendário
(data/raw/matches-schedule.csv) e os resultados reais
(data/raw/worldcup-2026-results.csv), reconcilia os schemas divergentes e
calcula métricas próprias para previsão de futebol (acurácia, Brier multiclasse,
log-loss, RPS) mais calibração e baselines. Grava a tabela jogo-a-jogo, o resumo
(JSON + Markdown) e o diagrama de calibração em artifacts/.

A definição e a interpretação das métricas estão em ../references/metricas.md.

Uso típico, a partir da raiz do repositório:

    python .claude/skills/avaliar-previsoes/scripts/score_predictions.py

Só são pontuadas as partidas que já têm resultado real — durante a fase de
grupos isso é um subconjunto do calendário, o que é esperado.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Ordem ordinal das saídas: vitória do mandante -> empate -> vitória do visitante.
# Essa ordem importa para o RPS, que penaliza erros "distantes" mais que os "vizinhos".
OUTCOMES = ("home", "draw", "away")
PROB_COLS = ("p_home", "p_draw", "p_away")
EPS = 1e-15

# Variantes de nome conhecidas -> forma canônica usada no calendário/resultados.
# Útil sobretudo para casar com o ranking FIFA (country_full), que usa a grafia da FIFA.
DEFAULT_ALIASES = {
    "united states": "usa",
    "united states of america": "usa",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "ir iran": "iran",
    "iran islamic republic of": "iran",
    "cote d'ivoire": "ivory coast",
    "cote d ivoire": "ivory coast",
    "czechia": "czech republic",
    "turkiye": "turkey",
    "cabo verde": "cape verde",
    "china pr": "china",
    "the netherlands": "netherlands",
    "bosnia": "bosnia and herzegovina",
}


class ScoringError(RuntimeError):
    """Erro ao pontuar as previsões (entrada ausente ou inconsistente)."""


# --------------------------------------------------------------------------- #
# Localização e leitura
# --------------------------------------------------------------------------- #
def find_repo_root() -> Path:
    """Sobe a partir do CWD procurando o calendário; cai para a posição da skill."""
    here = Path.cwd()
    for base in (here, *here.parents):
        if (base / "data" / "raw" / "matches-schedule.csv").is_file():
            return base
    # .claude/skills/avaliar-previsoes/scripts/score_predictions.py -> repo na 4ª subida
    return Path(__file__).resolve().parents[4]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ScoringError(f"Arquivo ausente: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _norm(name: object) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _canon(name: object, aliases: dict[str, str]) -> str:
    key = _norm(name)
    return aliases.get(key, key)


# --------------------------------------------------------------------------- #
# Métricas (ver references/metricas.md para as fórmulas)
# --------------------------------------------------------------------------- #
def _onehot(outcome: str) -> np.ndarray:
    return np.array([1.0 if o == outcome else 0.0 for o in OUTCOMES])


def brier_multiclass(probs: np.ndarray, onehots: np.ndarray) -> float:
    """Brier multiclasse: média de sum_k (p_k - o_k)^2. Intervalo [0, 2]."""
    return float(np.mean(np.sum((probs - onehots) ** 2, axis=1)))


def log_loss_mean(probs: np.ndarray, onehots: np.ndarray) -> float:
    """Log-loss média: -log(p_correto), com clipping para evitar log(0)."""
    clipped = np.clip(probs, EPS, 1.0)
    return float(np.mean(-np.sum(onehots * np.log(clipped), axis=1)))


def rps_mean(probs: np.ndarray, onehots: np.ndarray) -> float:
    """Ranked Probability Score médio sobre saídas ordinais. Intervalo [0, 1]."""
    cdf_p = np.cumsum(probs, axis=1)
    cdf_o = np.cumsum(onehots, axis=1)
    r = probs.shape[1]
    return float(np.mean(np.sum((cdf_p[:, :-1] - cdf_o[:, :-1]) ** 2, axis=1) / (r - 1)))


def accuracy(probs: np.ndarray, actual_idx: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == actual_idx))


def reliability_bins(
    probs: np.ndarray, onehots: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confiabilidade one-vs-rest agrupada: prob. média prevista x frequência observada."""
    p = probs.ravel()
    o = onehots.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        mean_pred.append(float(p[mask].mean()))
        obs_freq.append(float(o[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


# --------------------------------------------------------------------------- #
# Montagem do conjunto pontuável
# --------------------------------------------------------------------------- #
def _validate_and_normalize_probs(preds: pd.DataFrame) -> pd.DataFrame:
    missing = ({"match", *PROB_COLS}) - set(preds.columns)
    if missing:
        raise ScoringError(
            "predictions_submission.csv: colunas ausentes: "
            + ", ".join(sorted(missing))
            + ". Contrato esperado: match, home, away, p_home, p_draw, p_away."
        )
    probs = preds[list(PROB_COLS)].to_numpy(dtype=float)
    if np.isnan(probs).any():
        raise ScoringError("Há probabilidades vazias/NaN em predictions_submission.csv.")
    if (probs < 0).any():
        raise ScoringError("Há probabilidades negativas em predictions_submission.csv.")
    sums = probs.sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 1e-3):
        n_off = int(np.sum(np.abs(sums - 1.0) > 1e-3))
        print(
            f"AVISO: {n_off} previsão(ões) não somam 1; renormalizando.",
            file=sys.stderr,
        )
        probs = probs / sums[:, None]
        preds = preds.copy()
        preds[list(PROB_COLS)] = probs
    return preds


def build_full_frame(
    preds: pd.DataFrame,
    schedule: pd.DataFrame,
    results: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    """Junta previsões -> calendário (por match) -> resultados (left join pelo par).

    Mantém TODAS as previsões; nas partidas ainda sem resultado, `actual`,
    `home_score`, `away_score` e `result_date` ficam NaN. O subconjunto pontuável
    é `frame[frame["actual"].notna()]` — é o que alimenta as métricas; o quadro
    completo alimenta a comparação legível (Markdown) das 72 partidas.
    """
    sched = schedule.copy()
    sched["match"] = sched["match"].astype(int)
    sched["k_home"] = sched["home"].map(lambda x: _canon(x, aliases))
    sched["k_away"] = sched["away"].map(lambda x: _canon(x, aliases))

    res = results.copy()
    res["k_home"] = res["home_team"].map(lambda x: _canon(x, aliases))
    res["k_away"] = res["away_team"].map(lambda x: _canon(x, aliases))
    res["actual"] = np.where(
        res["home_score"] > res["away_score"],
        "home",
        np.where(res["home_score"] < res["away_score"], "away", "draw"),
    )
    res = res.rename(columns={"date": "result_date"})
    res = res[["k_home", "k_away", "result_date", "home_score", "away_score", "actual"]]

    preds = preds.copy()
    preds["match"] = preds["match"].astype(int)

    merged = preds.merge(
        sched[["match", "date", "home", "away", "k_home", "k_away"]],
        on="match",
        how="left",
        suffixes=("", "_sched"),
    )
    if merged[["k_home", "k_away"]].isna().any().any():
        orfas = merged.loc[merged["k_home"].isna(), "match"].tolist()
        raise ScoringError(
            f"Previsões sem partida correspondente no calendário (match): {orfas}"
        )

    merged = merged.merge(res, on=["k_home", "k_away"], how="left")
    return merged.sort_values("match").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def base_rate_probs(historical: pd.DataFrame) -> np.ndarray | None:
    """Frequências históricas de 1X2 (mandante/empate/visitante) como vetor único."""
    try:
        hs = pd.to_numeric(historical["home_score"], errors="coerce")
        as_ = pd.to_numeric(historical["away_score"], errors="coerce")
    except KeyError:
        return None
    mask = hs.notna() & as_.notna()
    hs, as_ = hs[mask], as_[mask]
    if len(hs) == 0:
        return None
    n = len(hs)
    return np.array(
        [(hs > as_).sum() / n, (hs == as_).sum() / n, (hs < as_).sum() / n]
    )


def ranking_accuracy(
    scored: pd.DataFrame, ranking: pd.DataFrame, aliases: dict[str, str]
) -> tuple[float, int] | None:
    """Acurácia do baseline 'maior ranking FIFA vence' sobre as partidas pontuadas."""
    try:
        rk = ranking.copy()
        rk["rank_date"] = pd.to_datetime(rk["rank_date"], errors="coerce")
        rk = rk.sort_values("rank_date").dropna(subset=["rank_date"])
        rk["k"] = rk["country_full"].map(lambda x: _canon(x, aliases))
        latest = rk.groupby("k")["rank"].last()
    except KeyError:
        return None

    hits = total = 0
    for _, row in scored.iterrows():
        rh, ra = latest.get(row["k_home"]), latest.get(row["k_away"])
        if pd.isna(rh) or pd.isna(ra) or rh == ra:
            continue
        pick = "home" if rh < ra else "away"  # rank menor = melhor colocado
        total += 1
        hits += int(pick == row["actual"])
    if total == 0:
        return None
    return hits / total, total


# --------------------------------------------------------------------------- #
# Saídas
# --------------------------------------------------------------------------- #
def write_calibration_plot(probs, onehots, out_path: Path, n_scored: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_pred, obs_freq, counts = reliability_bins(probs, onehots)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
    if len(mean_pred):
        sizes = 30 + 8 * counts
        ax.scatter(mean_pred, obs_freq, s=sizes, color="#1f77b4", zorder=3)
        ax.plot(mean_pred, obs_freq, color="#1f77b4", alpha=0.6, label="modelo")
    ax.set_xlabel("Probabilidade prevista (one-vs-rest)")
    ax.set_ylabel("Frequência observada")
    ax.set_title(f"Diagrama de calibração ({n_scored} partidas pontuadas)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_summary_md(summary: dict) -> str:
    m = summary["modelo"]
    lines = [
        "# Avaliação das previsões — Paul the Octopus",
        "",
        f"- Previsões na submissão: **{summary['n_previsoes']}**",
        f"- Partidas pontuadas (já com resultado): **{summary['n_pontuadas']}**",
        "",
        "## Métricas do modelo",
        "",
        "| Métrica | Valor | Direção |",
        "|---|---|---|",
        f"| Acurácia | {m['acuracia']:.3f} | ↑ melhor |",
        f"| Brier (multiclasse) | {m['brier']:.4f} | ↓ melhor |",
        f"| Log-loss | {m['log_loss']:.4f} | ↓ melhor |",
        f"| RPS | {m['rps']:.4f} | ↓ melhor |",
    ]
    if "acerto_exato_placar" in m:
        lines.append(
            f"| Acerto exato de placar | {m['acerto_exato_placar']:.3f} "
            f"({m['n_placar_exato']}/{summary['n_pontuadas']}) | ↑ melhor |"
        )
    lines += [
        "",
        "## Baselines",
        "",
        "| Baseline | Métrica | Valor |",
        "|---|---|---|",
    ]
    b = summary["baselines"]
    if "taxa_base" in b:
        tb = b["taxa_base"]
        lines.append(f"| Taxa-base histórica | Brier / log-loss / RPS "
                     f"| {tb['brier']:.4f} / {tb['log_loss']:.4f} / {tb['rps']:.4f} |")
    if "sempre_mandante" in b:
        lines.append(f"| Sempre mandante | Acurácia | {b['sempre_mandante']:.3f} |")
    if "maior_ranking" in b:
        mr = b["maior_ranking"]
        lines.append(f"| Maior ranking FIFA | Acurácia ({mr['n']} jogos) "
                     f"| {mr['acuracia']:.3f} |")
    lines += [
        "",
        "## Arquivos",
        "",
        f"- Tabela jogo-a-jogo: `{summary['arquivos']['jogo_a_jogo']}`",
        f"- Calibração: `{summary['arquivos']['calibracao']}`",
        "",
        "_RPS e Brier menores são melhores; o teste de fogo é **superar a "
        "taxa-base** e bater os palpites triviais. Com poucas partidas, leia a "
        "calibração com cautela._",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def score(args: argparse.Namespace) -> dict:
    aliases = dict(DEFAULT_ALIASES)
    if args.aliases:
        aliases.update(
            {_norm(k): _norm(v) for k, v in json.loads(Path(args.aliases).read_text()).items()}
        )

    preds = _validate_and_normalize_probs(_read_csv(args.predictions))
    schedule = _read_csv(args.schedule)
    results = _read_csv(args.results)

    scored = build_scored_frame(preds, schedule, results, aliases)
    if scored.empty:
        raise ScoringError(
            "Nenhuma partida pôde ser pontuada: não há interseção entre as previsões "
            "e os resultados reais (verifique nomes de seleções e o calendário)."
        )

    probs = scored[list(PROB_COLS)].to_numpy(dtype=float)
    onehots = np.vstack([_onehot(o) for o in scored["actual"]])
    actual_idx = np.array([OUTCOMES.index(o) for o in scored["actual"]])

    # Tabela jogo-a-jogo
    scored["previsto"] = [OUTCOMES[i] for i in np.argmax(probs, axis=1)]
    scored["acertou"] = (scored["previsto"] == scored["actual"]).astype(int)
    scored["brier"] = np.sum((probs - onehots) ** 2, axis=1)
    scored["log_loss"] = -np.sum(onehots * np.log(np.clip(probs, EPS, 1.0)), axis=1)
    cdf_p, cdf_o = np.cumsum(probs, axis=1), np.cumsum(onehots, axis=1)
    scored["rps"] = np.sum((cdf_p[:, :-1] - cdf_o[:, :-1]) ** 2, axis=1) / (len(OUTCOMES) - 1)

    # Componente de placar COMPANHEIRO (retrocompatível): só se a submissão trouxer
    # 'placar_provavel'. Compara placar previsto x real e mede o acerto EXATO de
    # placar. Não altera nada do 1X2 acima nem o contrato/baselines.
    has_scoreline = "placar_provavel" in scored.columns and scored["placar_provavel"].notna().any()
    score_extra_cols: list[str] = []
    if has_scoreline:
        scored["placar_real"] = (
            scored["home_score"].astype("Int64").astype(str)
            + "-"
            + scored["away_score"].astype("Int64").astype(str)
        )
        scored["placar_acertou"] = (
            scored["placar_provavel"].astype(str) == scored["placar_real"]
        ).astype(int)
        score_extra_cols = ["placar_provavel", "placar_real", "placar_acertou"]
        if "xg_home" in scored.columns:
            score_extra_cols = ["xg_home", "xg_away", *score_extra_cols]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["match", "date", "home", "away", *PROB_COLS,
            "home_score", "away_score", "previsto", "actual", "acertou",
            *score_extra_cols, "brier", "log_loss", "rps"]
    jogo_a_jogo = out_dir / "avaliacao_jogo_a_jogo.csv"
    scored[cols].to_csv(jogo_a_jogo, index=False)

    calib = out_dir / "calibracao.png"
    write_calibration_plot(probs, onehots, calib, len(scored))

    # Baselines
    baselines: dict = {}
    br = base_rate_probs(_read_csv(args.historical)) if Path(args.historical).is_file() else None
    if br is not None:
        bp = np.tile(br, (len(scored), 1))
        baselines["taxa_base"] = {
            "probs": br.round(4).tolist(),
            "brier": brier_multiclass(bp, onehots),
            "log_loss": log_loss_mean(bp, onehots),
            "rps": rps_mean(bp, onehots),
        }
    baselines["sempre_mandante"] = float(np.mean(scored["actual"] == "home"))
    if Path(args.ranking).is_file():
        rk = ranking_accuracy(scored, _read_csv(args.ranking), aliases)
        if rk is not None:
            baselines["maior_ranking"] = {"acuracia": rk[0], "n": rk[1]}

    summary = {
        "n_previsoes": int(len(preds)),
        "n_pontuadas": int(len(scored)),
        "modelo": {
            "acuracia": accuracy(probs, actual_idx),
            "brier": brier_multiclass(probs, onehots),
            "log_loss": log_loss_mean(probs, onehots),
            "rps": rps_mean(probs, onehots),
        },
        "baselines": baselines,
        "arquivos": {
            "jogo_a_jogo": str(jogo_a_jogo),
            "calibracao": str(calib),
            "resumo_json": str(out_dir / "avaliacao_resumo.json"),
            "resumo_md": str(out_dir / "avaliacao_resumo.md"),
        },
    }
    # Acerto exato de placar (companheiro): % de jogos em que o placar previsto bate
    # o real. Só quando a submissão traz placar; senão ausente (retrocompatível).
    if has_scoreline:
        summary["modelo"]["acerto_exato_placar"] = float(scored["placar_acertou"].mean())
        summary["modelo"]["n_placar_exato"] = int(scored["placar_acertou"].sum())

    (out_dir / "avaliacao_resumo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    (out_dir / "avaliacao_resumo.md").write_text(render_summary_md(summary))
    return summary


def build_parser(repo: Path) -> argparse.ArgumentParser:
    raw, results_dir = repo / "data" / "raw", repo / "data" / "results"
    p = argparse.ArgumentParser(description="Pontua as previsões 1X2 contra os resultados reais.")
    p.add_argument("--predictions", type=Path, default=results_dir / "predictions_submission.csv")
    p.add_argument("--results", type=Path, default=raw / "worldcup-2026-results.csv")
    p.add_argument("--schedule", type=Path, default=raw / "matches-schedule.csv")
    p.add_argument("--ranking", type=Path, default=raw / "ranking.csv")
    p.add_argument("--historical", type=Path, default=raw / "historical-results.csv")
    p.add_argument("--out-dir", type=Path, default=repo / "artifacts")
    p.add_argument("--aliases", type=Path, default=None, help="JSON opcional {variante: canônico}.")
    return p


def main() -> int:
    args = build_parser(find_repo_root()).parse_args()
    try:
        summary = score(args)
    except ScoringError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    m = summary["modelo"]
    print(f"OK: {summary['n_pontuadas']}/{summary['n_previsoes']} partidas pontuadas")
    print(f"  acurácia={m['acuracia']:.3f}  Brier={m['brier']:.4f}  "
          f"log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}")
    if "acerto_exato_placar" in m:
        print(f"  acerto exato de placar={m['acerto_exato_placar']:.3f} "
              f"({m['n_placar_exato']}/{summary['n_pontuadas']})")
    tb = summary["baselines"].get("taxa_base")
    if tb:
        print(f"  taxa-base: Brier={tb['brier']:.4f}  log-loss={tb['log_loss']:.4f}  "
              f"RPS={tb['rps']:.4f}")
    print(f"  resumo -> {summary['arquivos']['resumo_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
