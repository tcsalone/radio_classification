"""Text-derived commercial identity (no audio fingerprints for ads)."""

from radio_classifier.commercials.dedupe import (
    CommercialDedupeGroup,
    CommercialDedupeMember,
    CommercialDedupeReport,
    dedupe_commercials,
)
from radio_classifier.commercials.identity import (
    CommercialIdentityResolver,
    CommercialResolution,
    CommercialResolverConfig,
    bucket_duration,
)

__all__ = [
    "CommercialDedupeGroup",
    "CommercialDedupeMember",
    "CommercialDedupeReport",
    "CommercialIdentityResolver",
    "CommercialResolution",
    "CommercialResolverConfig",
    "bucket_duration",
    "dedupe_commercials",
]
