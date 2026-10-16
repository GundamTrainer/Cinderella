from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace

from integrations.interfaces import ProviderExecutionContext
from integrations.network import ProviderRequestError
from integrations.policies import NetworkMode, ProviderPolicyError
from integrations.providers.knowledge import MediaWikiProvider
from integrations.providers.weather import OpenMeteoProvider
from runtime.config import ConfigurationError, load_runtime_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly opt in to a single provider smoke check."
    )
    parser.add_argument("--provider", choices=("mediawiki", "open-meteo"), required=True)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicitly permit one request if provider policy validation passes.",
    )
    parser.add_argument("--confirm-query-text", action="store_true")
    parser.add_argument("--confirm-location-transfer", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--timezone", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_network:
        print("Blocked: pass --allow-network for an explicit one-request check.", file=sys.stderr)
        return 2
    if args.provider == "mediawiki":
        if not args.query or not args.confirm_query_text:
            print(
                "Blocked: --query and --confirm-query-text are required.",
                file=sys.stderr,
            )
            return 2
    elif (
        args.latitude is None
        or args.longitude is None
        or not args.confirm_location_transfer
    ):
        print(
            "Blocked: explicit coordinates and --confirm-location-transfer are required.",
            file=sys.stderr,
        )
        return 2

    try:
        settings = load_runtime_settings()
        policy = replace(
            settings.network,
            mode=NetworkMode.ASK_BEFORE_EACH_PROVIDER,
            allow_external_requests=True,
        )
        if args.provider == "mediawiki":
            provider = MediaWikiProvider(policy=policy, enabled=True)
            arguments = {"query": args.query}
            confirmed = args.confirm_query_text
        else:
            provider = OpenMeteoProvider(policy=policy, enabled=True)
            arguments = {
                "latitude": args.latitude,
                "longitude": args.longitude,
                "timezone": args.timezone,
            }
            confirmed = args.confirm_location_transfer
        provider.initialize()
        result = provider.execute(
            arguments,
            context=ProviderExecutionContext(
                user_confirmed=confirmed,
                user_data_included=True,
            ),
        )
    except (ConfigurationError, ProviderPolicyError) as exc:
        print(f"Provider check blocked by configuration or policy: {exc}", file=sys.stderr)
        return 2
    except ProviderRequestError as exc:
        print(f"Provider request failed: {exc}", file=sys.stderr)
        return 3
    finally:
        if "provider" in locals():
            provider.close()

    print(
        json.dumps(
            {
                "provider_id": result.provider_id,
                "retrieved_at": result.retrieved_at,
                "from_cache": result.from_cache,
                "data": result.data,
                "sources": [asdict(source) for source in result.sources],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
