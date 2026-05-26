"""Text-derived commercial identity (no audio fingerprints for ads)."""

from radio_classifier.commercials.identity import (
    CommercialIdentityResolver,
    CommercialResolution,
    CommercialResolverConfig,
    bucket_duration,
)

__all__ = [
    "CommercialIdentityResolver",
    "CommercialResolution",
    "CommercialResolverConfig",
    "bucket_duration",
]
