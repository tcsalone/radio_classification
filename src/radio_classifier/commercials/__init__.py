"""Text-derived commercial identity (no audio fingerprints for ads)."""

from radio_classifier.commercials.backfill import (
    BrandBackfillItem,
    BrandBackfillReport,
    backfill_unbranded_commercials,
)
from radio_classifier.commercials.brand_extract import extract_brand_from_text
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
from radio_classifier.commercials.merge_boundaries import (
    BoundaryMergeItem,
    BoundaryMergeReport,
    merge_boundary_commercials,
)

__all__ = [
    "BoundaryMergeItem",
    "BoundaryMergeReport",
    "BrandBackfillItem",
    "BrandBackfillReport",
    "CommercialDedupeGroup",
    "CommercialDedupeMember",
    "CommercialDedupeReport",
    "CommercialIdentityResolver",
    "CommercialResolution",
    "CommercialResolverConfig",
    "backfill_unbranded_commercials",
    "bucket_duration",
    "dedupe_commercials",
    "extract_brand_from_text",
    "merge_boundary_commercials",
]
