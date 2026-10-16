from __future__ import annotations

import hashlib
import json
import urllib.parse
from datetime import UTC, datetime, timedelta
from typing import Any

from integrations.cache import MemoryCache
from integrations.interfaces import (
    ProviderClarificationRequired,
    ProviderExecutionContext,
    ProviderInfo,
    ProviderResult,
    SourceRecord,
)
from integrations.network import JsonTransport, UrllibJsonTransport
from integrations.policies import NetworkPolicy, ProviderPolicyError

from .base import CachedNetworkProvider

REVIEW_DATE = "2026-10-16"
OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


class OpenMeteoProvider(CachedNetworkProvider):
    """Fetches current weather only for caller-supplied coordinates; never geolocates."""

    def __init__(
        self,
        *,
        policy: NetworkPolicy,
        enabled: bool = False,
        transport: JsonTransport | None = None,
        cache: MemoryCache[ProviderResult] | None = None,
    ) -> None:
        endpoint = OPEN_METEO_ENDPOINT
        policy.validate_url(endpoint)
        self._endpoint = endpoint
        info = ProviderInfo(
            provider_id="open-meteo-weather",
            provider_name="Open-Meteo weather API",
            provider_version="Forecast API",
            provider_type="weather",
            base_url=endpoint,
            authentication_type="none",
            free_access=True,
            payment_possible=True,
            paid_quality_upgrade=True,
            credit_card_required=False,
            self_hostable=False,
            open_source=True,
            data_license="CC BY 4.0; attribution required",
            terms_url="https://open-meteo.com/en/terms",
            privacy_url=None,
            network_required=True,
            sends_user_content=True,
            sends_camera_data=False,
            sends_audio_data=False,
            rate_limit="Free API: 10,000 calls/day; commercial customer API is paid.",
            attribution_required=True,
            cache_allowed=True,
            last_policy_review=REVIEW_DATE,
            enabled=enabled,
            user_data_retention=(
                "Not confirmed; coordinates are query parameters and provider retention details "
                "require review before activation."
            ),
        )
        super().__init__(
            info,
            policy=policy,
            transport=transport or UrllibJsonTransport(policy),
            cache=cache,
            cache_ttl_seconds=600,
            minimum_request_interval_seconds=1.0,
        )

    def capabilities(self) -> tuple[str, ...]:
        return ("current_weather", "temperature", "precipitation", "wind", "humidity")

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ProviderExecutionContext | None = None,
    ) -> ProviderResult:
        latitude = arguments.get("latitude")
        longitude = arguments.get("longitude")
        if latitude is None or longitude is None:
            raise ProviderClarificationRequired(
                "Ask the user to provide or explicitly select a location before requesting weather."
            )
        if (
            isinstance(latitude, bool)
            or not isinstance(latitude, (int, float))
            or not -90 <= latitude <= 90
            or isinstance(longitude, bool)
            or not isinstance(longitude, (int, float))
            or not -180 <= longitude <= 180
        ):
            raise ValueError("Explicit latitude and longitude within valid ranges are required.")
        timezone = arguments.get("timezone", "auto")
        if not isinstance(timezone, str) or not timezone or len(timezone) > 64:
            raise ValueError("timezone must be a non-empty string of at most 64 characters.")
        units = arguments.get("temperature_unit", "celsius")
        if not isinstance(units, str) or units not in {"celsius", "fahrenheit"}:
            raise ValueError("temperature_unit must be 'celsius' or 'fahrenheit'.")
        forecast_days = arguments.get("forecast_days", 1)
        if (
            isinstance(forecast_days, bool)
            or not isinstance(forecast_days, int)
            or not 1 <= forecast_days <= 3
        ):
            raise ValueError("forecast_days must be an integer from 1 to 3.")
        normalized = {
            "latitude": round(float(latitude), 4),
            "longitude": round(float(longitude), 4),
            "timezone": timezone,
            "temperature_unit": units,
            "forecast_days": forecast_days,
        }
        cached = self._cached(normalized)
        if cached is not None:
            return cached
        if context is None or not context.user_confirmed:
            raise ProviderPolicyError(
                "Weather requests require confirmation to send the supplied location."
            )
        self._authorize(
            ProviderExecutionContext(
                user_confirmed=context.user_confirmed,
                user_data_included=True,
                contains_personal_data=context.contains_personal_data,
                development_test=context.development_test,
            )
        )
        params = urllib.parse.urlencode(
            {
                "latitude": normalized["latitude"],
                "longitude": normalized["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,rain,showers,snowfall,weather_code,wind_speed_10m,"
                    "wind_direction_10m"
                ),
                "temperature_unit": units,
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
                "timezone": timezone,
                "daily": (
                    "temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                    "precipitation_sum,wind_speed_10m_max,weather_code"
                ),
                "forecast_days": forecast_days,
            }
        )
        response = self._get_json(f"{self._endpoint}?{params}")
        weather, units_result, forecast, forecast_units = self.parse_current_response(response)
        location_label = arguments.get("location_label")
        if location_label is not None and (
            not isinstance(location_label, str) or len(location_label) > 128
        ):
            raise ValueError("location_label must be a string of at most 128 characters.")
        retrieved = datetime.now(UTC)
        ttl_expires = retrieved + timedelta(minutes=10)
        digest = hashlib.sha256(
            json.dumps(
                {"weather": weather, "forecast": forecast},
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        source = SourceRecord.create(
            provider_id=self.provider_info.provider_id,
            title="Open-Meteo current weather",
            canonical_url="https://open-meteo.com/",
            query="current weather for user-supplied coordinates",
            result_hash=digest,
            license="CC BY 4.0",
            attribution="Weather data by Open-Meteo and its data providers",
            cache_expiration=ttl_expires.isoformat(),
        )
        result = ProviderResult(
            provider_id=self.provider_info.provider_id,
            data={
                "weather": weather,
                "units": units_result,
                "forecast": forecast,
                "forecast_units": forecast_units,
                "timezone": response.get("timezone", timezone),
                "location_label": location_label,
                "coordinates_user_supplied": True,
                "source_kind": "external_source",
            },
            sources=(source,),
            retrieved_at=retrieved.isoformat(),
        )
        return self._store(normalized, result)

    @staticmethod
    def parse_current_response(
        response: dict[str, object],
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        current = response.get("current")
        current_units = response.get("current_units")
        daily = response.get("daily", {})
        daily_units = response.get("daily_units", {})
        if (
            not isinstance(current, dict)
            or not isinstance(current_units, dict)
            or not isinstance(daily, dict)
            or not isinstance(daily_units, dict)
        ):
            raise TypeError("Open-Meteo returned a malformed current-weather response.")
        allowed = {
            "time",
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weather_code",
            "wind_speed_10m",
            "wind_direction_10m",
        }
        weather = {key: current[key] for key in allowed if key in current}
        units_result = {key: current_units[key] for key in allowed if key in current_units}
        daily_allowed = {
            "time",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum",
            "wind_speed_10m_max",
            "weather_code",
        }
        forecast = {key: daily[key] for key in daily_allowed if key in daily}
        forecast_units = {
            key: daily_units[key] for key in daily_allowed if key in daily_units
        }
        return weather, units_result, forecast, forecast_units
