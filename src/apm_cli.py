from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from apm import (
    DEFAULT_END,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_START,
    Position,
    main as apm_main,
    market_hedge,
    portfolio_volatility,
    single_index_regression,
)

Holding = dict[str, object]


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _parse_position_value(value: str) -> Position:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) < 2:
        raise ValueError("Position must be 'notional,beta[,idiosyncratic_volatility]'.")
    notional = float(parts[0])
    beta = float(parts[1])
    idio = float(parts[2]) if len(parts) > 2 else 0.0
    return Position(notional=notional, beta=beta, idiosyncratic_volatility=idio)


def _positions_from_payload(payload: object) -> list[Position]:
    if isinstance(payload, dict) and "positions" in payload:
        payload = payload["positions"]
    if not isinstance(payload, list):
        raise ValueError("Positions payload must be a list.")

    positions: list[Position] = []
    for item in payload:
        if isinstance(item, dict):
            if "notional" not in item or "beta" not in item:
                raise ValueError("Each position must include notional and beta.")
            positions.append(
                Position(
                    notional=float(item["notional"]),
                    beta=float(item["beta"]),
                    idiosyncratic_volatility=float(item.get("idiosyncratic_volatility", 0.0)),
                )
            )
        elif isinstance(item, (list, tuple)):
            if len(item) < 2:
                raise ValueError("Each position must include notional and beta.")
            idio = float(item[2]) if len(item) > 2 else 0.0
            positions.append(Position(notional=float(item[0]), beta=float(item[1]), idiosyncratic_volatility=idio))
        else:
            raise ValueError("Positions must be objects or arrays.")
    return positions


def _positions_from_csv(path: Path) -> list[Position]:
    positions: list[Position] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("Positions CSV is empty.")

    header = rows[0]
    has_header = any(not _is_float(cell) for cell in header)

    if has_header:
        fieldnames = [name.strip() for name in rows[0]]
        for row in rows[1:]:
            if not row:
                continue
            row_map = {fieldnames[i]: row[i].strip() if i < len(row) else "" for i in range(len(fieldnames))}
            notional = row_map.get("notional")
            beta = row_map.get("beta")
            if notional is None or beta is None:
                raise ValueError("CSV header must include notional and beta columns.")
            idio = row_map.get("idiosyncratic_volatility", "0")
            positions.append(
                Position(
                    notional=float(notional),
                    beta=float(beta),
                    idiosyncratic_volatility=float(idio or 0.0),
                )
            )
    else:
        for row in rows:
            if not row:
                continue
            if len(row) < 2:
                raise ValueError("CSV rows must include notional and beta values.")
            idio = float(row[2]) if len(row) > 2 else 0.0
            positions.append(Position(notional=float(row[0]), beta=float(row[1]), idiosyncratic_volatility=idio))

    return positions


def _holdings_from_payload(payload: object) -> list[Holding]:
    if isinstance(payload, dict):
        if "holdings" in payload:
            payload = payload["holdings"]
        elif "positions" in payload:
            payload = payload["positions"]
    if not isinstance(payload, list):
        raise ValueError("Holdings payload must be a list.")

    holdings: list[Holding] = []
    for item in payload:
        if isinstance(item, dict):
            if "ticker" not in item or "notional" not in item:
                raise ValueError("Each holding must include ticker and notional.")
            holdings.append({"ticker": str(item["ticker"]), "notional": float(item["notional"])})
        elif isinstance(item, (list, tuple)):
            if len(item) < 2:
                raise ValueError("Each holding must include ticker and notional.")
            holdings.append({"ticker": str(item[0]), "notional": float(item[1])})
        else:
            raise ValueError("Holdings must be objects or arrays.")
    return holdings


def _holdings_from_csv(path: Path) -> list[Holding]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("Holdings CSV is empty.")

    header = [cell.strip().lower() for cell in rows[0]]
    has_header = "ticker" in header or "notional" in header

    holdings: list[Holding] = []
    if has_header:
        fieldnames = [cell.strip() for cell in rows[0]]
        for row in rows[1:]:
            if not row:
                continue
            row_map = {fieldnames[i]: row[i].strip() if i < len(row) else "" for i in range(len(fieldnames))}
            ticker = row_map.get("ticker")
            notional = row_map.get("notional")
            if ticker is None or notional is None:
                raise ValueError("CSV header must include ticker and notional columns.")
            holdings.append({"ticker": str(ticker), "notional": float(notional)})
    else:
        for row in rows:
            if not row:
                continue
            if len(row) < 2:
                raise ValueError("CSV rows must include ticker and notional values.")
            holdings.append({"ticker": str(row[0]).strip(), "notional": float(row[1])})

    return holdings


def _load_positions(
    *,
    positions_json: str | None,
    positions_file: str | None,
    positions_format: str | None,
    position_values: list[str] | None,
) -> list[Position]:
    sources = [
        bool(positions_json),
        bool(positions_file),
        bool(position_values),
    ]
    if sum(sources) == 0:
        raise ValueError("Provide positions via --position, --positions, or --positions-file.")
    if sum(sources) > 1:
        raise ValueError("Use only one positions source at a time.")

    if position_values:
        return [_parse_position_value(value) for value in position_values]

    if positions_json:
        payload = json.loads(positions_json)
        return _positions_from_payload(payload)

    if positions_file:
        path = Path(positions_file)
        fmt = positions_format
        if fmt is None or fmt == "auto":
            fmt = "csv" if path.suffix.lower() == ".csv" else "json"
        if fmt == "csv":
            return _positions_from_csv(path)
        if fmt == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _positions_from_payload(payload)
        raise ValueError("positions-format must be 'json', 'csv', or 'auto'.")

    raise ValueError("Positions input is invalid.")


def _load_holdings(
    *,
    holdings_json: str | None,
    holdings_file: str | None,
    holdings_format: str | None,
    holding_values: list[str] | None,
) -> list[Holding]:
    sources = [
        bool(holdings_json),
        bool(holdings_file),
        bool(holding_values),
    ]
    if sum(sources) == 0:
        raise ValueError("Provide holdings via --holding, --holdings, or --holdings-file.")
    if sum(sources) > 1:
        raise ValueError("Use only one holdings source at a time.")

    if holding_values:
        holdings: list[Holding] = []
        for value in holding_values:
            parts = [part.strip() for part in value.split(",")]
            if len(parts) < 2:
                raise ValueError("Holding must be 'ticker,notional'.")
            holdings.append({"ticker": parts[0], "notional": float(parts[1])})
        return holdings

    if holdings_json:
        payload = json.loads(holdings_json)
        return _holdings_from_payload(payload)

    if holdings_file:
        path = Path(holdings_file)
        fmt = holdings_format
        if fmt is None or fmt == "auto":
            fmt = "csv" if path.suffix.lower() == ".csv" else "json"
        if fmt == "csv":
            return _holdings_from_csv(path)
        if fmt == "json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _holdings_from_payload(payload)
        raise ValueError("holdings-format must be 'json', 'csv', or 'auto'.")

    raise ValueError("Holdings input is invalid.")


def _build_positions_from_holdings(
    holdings: list[Holding],
    *,
    benchmark: str,
    period: str,
    interval: str,
    start: str | None,
    end: str | None,
    plot: bool,
) -> tuple[list[Position], list[dict[str, object]]]:
    positions: list[Position] = []
    regressions: list[dict[str, object]] = []

    for holding in holdings:
        ticker = str(holding["ticker"])
        notional = float(holding["notional"])
        regression = single_index_regression(
            ticker,
            benchmark,
            period=period,
            interval=interval,
            start=start,
            end=end,
            plot=plot,
        )
        idio_key = f"{ticker}_daily_idiosyncratic_volatility"
        idio = float(regression[idio_key])
        beta = float(regression["beta"])
        positions.append(Position(notional=notional, beta=beta, idiosyncratic_volatility=idio))
        regressions.append(
            {
                "ticker": ticker,
                "notional": notional,
                "alpha": float(regression["alpha"]),
                "beta": beta,
                "r_squared": float(regression["r_squared"]),
                "idiosyncratic_volatility": idio,
                "observations": float(regression["observations"]),
            }
        )

    return positions, regressions


def _render_table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["<"] * len(headers)
    if len(aligns) != len(headers):
        raise ValueError("aligns must match headers length.")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("All rows must have the same length as headers.")

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(row: list[str]) -> str:
        parts: list[str] = []
        for i, cell in enumerate(row):
            align = aligns[i]
            if align == ">":
                parts.append(cell.rjust(widths[i]))
            elif align == "^":
                parts.append(cell.center(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return " | ".join(parts)

    header_line = fmt_row(headers)
    sep_line = "-+-".join("-" * w for w in widths)
    body = "\n".join(fmt_row(row) for row in rows) if rows else ""
    return "\n".join(line for line in (header_line, sep_line, body) if line)


def _print_output(data: object, *, output: str) -> None:
    if output == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    if output == "table":
        if isinstance(data, dict):
            rows = [[str(key), str(value)] for key, value in data.items()]
            print(_render_table(["Metric", "Value"], rows, aligns=["<", ">"]))
            return
    print(data)


def _handle_single_index(args: argparse.Namespace) -> None:
    results = single_index_regression(
        args.stock,
        args.benchmark,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
        plot=args.plot,
    )

    if args.output == "table":
        rows = [[key, f"{value:.6f}"] for key, value in results.items()]
        print(_render_table(["Metric", "Value"], rows, aligns=["<", ">"]))
    else:
        _print_output(results, output=args.output)


def _handle_portfolio_volatility(args: argparse.Namespace) -> None:
    positions = _load_positions(
        positions_json=args.positions,
        positions_file=args.positions_file,
        positions_format=args.positions_format,
        position_values=args.position,
    )
    results = portfolio_volatility(positions, market_volatility=args.market_volatility)
    if args.output == "table":
        rows = [[key, f"{value:.6f}"] for key, value in results.items()]
        print(_render_table(["Metric", "Value"], rows, aligns=["<", ">"]))
    else:
        _print_output(results, output=args.output)


def _handle_market_hedge(args: argparse.Namespace) -> None:
    positions = _load_positions(
        positions_json=args.positions,
        positions_file=args.positions_file,
        positions_format=args.positions_format,
        position_values=args.position,
    )
    results = market_hedge(positions)

    if args.output == "table":
        rows = []
        for key, value in results.items():
            if isinstance(value, list):
                rows.append([key, ", ".join(f"{item:.6f}" for item in value)])
            else:
                rows.append([key, f"{float(value):.6f}"])
        print(_render_table(["Metric", "Value"], rows, aligns=["<", ">"]))
    else:
        _print_output(results, output=args.output)


def _handle_workflow(args: argparse.Namespace) -> None:
    holdings = _load_holdings(
        holdings_json=args.holdings,
        holdings_file=args.holdings_file,
        holdings_format=args.holdings_format,
        holding_values=args.holding,
    )
    positions, regressions = _build_positions_from_holdings(
        holdings,
        benchmark=args.benchmark,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
        plot=args.plot,
    )

    volatility = portfolio_volatility(positions, market_volatility=args.market_volatility)
    hedge = market_hedge(positions) if args.hedge else None

    if args.output == "json":
        payload: dict[str, object] = {
            "benchmark": args.benchmark,
            "regressions": regressions,
            "volatility": volatility,
        }
        if hedge is not None:
            payload["hedge"] = hedge
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    regression_rows = [
        [
            row["ticker"],
            f"{row['notional']:.2f}",
            f"{row['alpha']:.6f}",
            f"{row['beta']:.6f}",
            f"{row['r_squared']:.6f}",
            f"{row['idiosyncratic_volatility']:.6f}",
            f"{row['observations']:.0f}",
        ]
        for row in regressions
    ]
    print(
        _render_table(
            ["Ticker", "Notional", "Alpha", "Beta", "R^2", "Idio vol", "Obs"],
            regression_rows,
            aligns=["<", ">", ">", ">", ">", ">", ">"],
        )
    )

    volatility_rows = [[key, f"{value:.6f}"] for key, value in volatility.items()]
    print()
    print(_render_table(["Metric", "Value"], volatility_rows, aligns=["<", ">"]))

    if hedge is not None:
        hedge_rows = []
        for key, value in hedge.items():
            if isinstance(value, list):
                hedge_rows.append([key, ", ".join(f"{item:.6f}" for item in value)])
            else:
                hedge_rows.append([key, f"{float(value):.6f}"])
        print()
        print(_render_table(["Metric", "Value"], hedge_rows, aligns=["<", ">"]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apm-cli",
        description="CLI helpers for apm.py single-index regression and portfolio analytics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single-index", help="Run a single-index regression.")
    single.add_argument("stock", help="Stock ticker (e.g., NVDA)")
    single.add_argument("benchmark", help="Benchmark ticker (e.g., SPY)")
    single.add_argument("--period", default=DEFAULT_PERIOD)
    single.add_argument("--interval", default=DEFAULT_INTERVAL)
    single.add_argument("--start", default=DEFAULT_START)
    single.add_argument("--end", default=DEFAULT_END)
    single.add_argument("--plot", action="store_true", help="Show the regression plot.")
    single.add_argument("--output", choices=["table", "json"], default="table")
    single.set_defaults(func=_handle_single_index)

    portfolio = subparsers.add_parser(
        "portfolio-volatility", help="Compute portfolio volatility decomposition."
    )
    _add_positions_args(portfolio)
    portfolio.add_argument("--market-volatility", type=float, required=True)
    portfolio.add_argument("--output", choices=["table", "json"], default="table")
    portfolio.set_defaults(func=_handle_portfolio_volatility)

    hedge = subparsers.add_parser("market-hedge", help="Compute market hedge notional.")
    _add_positions_args(hedge)
    hedge.add_argument("--output", choices=["table", "json"], default="table")
    hedge.set_defaults(func=_handle_market_hedge)

    workflow = subparsers.add_parser(
        "workflow",
        help="Run holdings -> regressions -> volatility -> optional hedge workflow.",
    )
    _add_holdings_args(workflow)
    workflow.add_argument("benchmark", help="Benchmark ticker (e.g., SPY)")
    workflow.add_argument("--period", default=DEFAULT_PERIOD)
    workflow.add_argument("--interval", default=DEFAULT_INTERVAL)
    workflow.add_argument("--start", default=DEFAULT_START)
    workflow.add_argument("--end", default=DEFAULT_END)
    workflow.add_argument("--plot", action="store_true", help="Show regression plots.")
    workflow.add_argument("--market-volatility", type=float, required=True)
    workflow.add_argument("--hedge", action="store_true", help="Compute market hedge.")
    workflow.add_argument("--output", choices=["table", "json"], default="table")
    workflow.set_defaults(func=_handle_workflow)

    demo = subparsers.add_parser("demo", help="Run the apm.py demo workflow.")
    demo.set_defaults(func=lambda _args: apm_main())

    return parser


def _add_positions_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--position",
        action="append",
        help="Position as 'notional,beta[,idiosyncratic_volatility]'.",
    )
    parser.add_argument(
        "--positions",
        help="JSON string with positions (list of objects with notional/beta).",
    )
    parser.add_argument(
        "--positions-file",
        help="Path to JSON/CSV file with positions.",
    )
    parser.add_argument(
        "--positions-format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Force positions file format (default: auto by extension).",
    )


def _add_holdings_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--holding",
        action="append",
        help="Holding as 'ticker,notional'.",
    )
    parser.add_argument(
        "--holdings",
        help="JSON string with holdings (list of objects with ticker/notional).",
    )
    parser.add_argument(
        "--holdings-file",
        help="Path to JSON/CSV file with holdings.",
    )
    parser.add_argument(
        "--holdings-format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="Force holdings file format (default: auto by extension).",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
