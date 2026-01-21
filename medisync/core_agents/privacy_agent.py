"""
Privacy Module with K-Anonymity Filters

Implements privacy-preserving transformations for global insights:
- K-anonymity (minimum sample sizes)
- PII removal
- Generalization hierarchies
- Outlier suppression
"""

import re
import logging
import hashlib
from typing import Dict, Any, List, Optional, Set
from collections import Counter

logger = logging.getLogger(__name__)


class PrivacyFilter:
    """Privacy filters for global insights"""

    # Identifiable patterns to remove
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
        (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),  # Phone
        (r'\b[A-Z]\d{5,7}\b', '[ID]'),  # Patient/Doctor IDs
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
        (r'\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr)\b', '[ADDRESS]', re.IGNORECASE),  # Address
    ]

    # Generalization hierarchies
    AGE_BRACKETS = [
        (0, 18, '0-18'),
        (18, 30, '18-30'),
        (30, 40, '30-40'),
        (40, 50, '40-50'),
        (50, 60, '50-60'),
        (60, 70, '60-70'),
        (70, 120, '70+')
    ]

    BODY_PARTS_GENERALIZATION = {
        # Specific → General
        'right_index_finger': 'finger',
        'left_index_finger': 'finger',
        'right_thumb': 'finger',
        'left_thumb': 'finger',
        'right_hand': 'hand',
        'left_hand': 'hand',
        'right_foot': 'foot',
        'left_foot': 'foot',
        'right_knee': 'knee',
        'left_knee': 'knee',
        'upper_arm': 'arm',
        'lower_arm': 'arm',
        'forearm': 'arm'
    }

    @staticmethod
    def remove_pii(text: str) -> str:
        """
        Remove personally identifiable information from text

        Args:
            text: Raw text

        Returns:
            Text with PII removed
        """
        sanitized = text

        for pattern, replacement, *flags in PrivacyFilter.PII_PATTERNS:
            regex_flags = flags[0] if flags else 0
            sanitized = re.sub(pattern, replacement, sanitized, flags=regex_flags)

        return sanitized

    @staticmethod
    def generalize_age(age: int) -> str:
        """
        Generalize age to bracket

        Args:
            age: Exact age

        Returns:
            Age bracket string
        """
        for min_age, max_age, bracket in PrivacyFilter.AGE_BRACKETS:
            if min_age <= age < max_age:
                return bracket

        return '70+'

    @staticmethod
    def generalize_body_part(specific_part: str) -> str:
        """
        Generalize specific body part to general category

        Args:
            specific_part: Specific body part

        Returns:
            Generalized body part
        """
        normalized = specific_part.lower().replace(' ', '_')
        return PrivacyFilter.BODY_PARTS_GENERALIZATION.get(normalized, specific_part)

    @staticmethod
    def apply_k_anonymity(
        records: List[Dict[str, Any]],
        k: int = 20,
        min_clinics: int = 5,
        grouping_keys: List[str] = ['condition', 'treatment']
    ) -> List[Dict[str, Any]]:
        """
        Filter records to satisfy K-anonymity

        Args:
            records: List of records to filter
            k: Minimum records per group (K-anonymity parameter)
            min_clinics: Minimum number of contributing clinics
            grouping_keys: Keys to group by

        Returns:
            Filtered records satisfying K-anonymity
        """
        # Group records
        groups = {}
        for record in records:
            # Create group key
            group_key = tuple(record.get(key, '') for key in grouping_keys)

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append(record)

        # Filter groups
        filtered_records = []
        for group_key, group_records in groups.items():
            # Check K-anonymity
            if len(group_records) < k:
                logger.debug(
                    f"Group {group_key} has {len(group_records)} records (< {k}), "
                    "dropping for K-anonymity"
                )
                continue

            # Check clinic diversity
            clinic_ids = set(r.get('clinic_id', '') for r in group_records)
            if len(clinic_ids) < min_clinics:
                logger.debug(
                    f"Group {group_key} has {len(clinic_ids)} clinics (< {min_clinics}), "
                    "dropping for diversity"
                )
                continue

            filtered_records.extend(group_records)

        logger.info(
            f"K-anonymity filter: {len(records)} → {len(filtered_records)} records "
            f"(K={k}, min_clinics={min_clinics})"
        )

        return filtered_records

    @staticmethod
    def suppress_outliers(
        values: List[float],
        percentile: float = 5.0
    ) -> List[float]:
        """
        Suppress outliers (top and bottom percentiles)

        Args:
            values: List of values
            percentile: Percentile to suppress (e.g., 5 = top/bottom 5%)

        Returns:
            Values with outliers removed
        """
        if not values:
            return values

        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate cutoff indices
        lower_idx = int(n * percentile / 100)
        upper_idx = int(n * (100 - percentile) / 100)

        # Filter outliers
        filtered = sorted_values[lower_idx:upper_idx]

        logger.debug(
            f"Outlier suppression: {len(values)} → {len(filtered)} values "
            f"(removed top/bottom {percentile}%)"
        )

        return filtered

    @staticmethod
    def anonymize_record(
        record: Dict[str, Any],
        remove_keys: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Anonymize a single record

        Args:
            record: Record to anonymize
            remove_keys: Keys to remove (e.g., patient_id, clinic_id)

        Returns:
            Anonymized record
        """
        if remove_keys is None:
            remove_keys = {
                'patient_id', 'clinic_id', 'doctor_id',
                'id', 'timestamp', 'created_at'
            }

        anonymized = {}

        for key, value in record.items():
            # Remove specified keys
            if key in remove_keys:
                continue

            # Remove PII from text fields
            if isinstance(value, str):
                value = PrivacyFilter.remove_pii(value)

            # Generalize age
            if key == 'age' and isinstance(value, (int, float)):
                value = PrivacyFilter.generalize_age(int(value))

            # Generalize body parts
            if key in ['body_part', 'location'] and isinstance(value, str):
                value = PrivacyFilter.generalize_body_part(value)

            anonymized[key] = value

        return anonymized

    @staticmethod
    def aggregate_statistics(
        records: List[Dict[str, Any]],
        numeric_fields: List[str],
        categorical_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Aggregate records into statistics

        Args:
            records: List of records
            numeric_fields: Fields to aggregate numerically
            categorical_fields: Fields to aggregate categorically

        Returns:
            Aggregated statistics
        """
        import numpy as np

        stats = {
            'sample_size': len(records),
            'clinic_count': len(set(r.get('clinic_id', '') for r in records))
        }

        # Numeric aggregations
        for field in numeric_fields:
            values = [r[field] for r in records if field in r and r[field] is not None]

            if values:
                # Suppress outliers
                filtered_values = PrivacyFilter.suppress_outliers(values)

                if filtered_values:
                    stats[f'{field}_median'] = float(np.median(filtered_values))
                    stats[f'{field}_mean'] = float(np.mean(filtered_values))
                    stats[f'{field}_q25'] = float(np.percentile(filtered_values, 25))
                    stats[f'{field}_q75'] = float(np.percentile(filtered_values, 75))

        # Categorical aggregations
        for field in categorical_fields:
            values = [r[field] for r in records if field in r and r[field]]

            if values:
                counter = Counter(values)
                total = len(values)

                # Distribution as percentages
                distribution = {
                    value: round(count / total, 3)
                    for value, count in counter.items()
                }

                stats[f'{field}_distribution'] = distribution

        return stats

    @staticmethod
    def hash_identifier(identifier: str) -> str:
        """
        Hash an identifier for anonymization

        Args:
            identifier: Identifier to hash

        Returns:
            SHA256 hash (first 16 characters)
        """
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


class PrivacyValidator:
    """Validator for privacy compliance"""

    @staticmethod
    def validate_k_anonymity(
        records: List[Dict[str, Any]],
        k: int = 20,
        grouping_keys: List[str] = ['condition', 'treatment']
    ) -> bool:
        """
        Validate that records satisfy K-anonymity

        Args:
            records: Records to validate
            k: K-anonymity parameter
            grouping_keys: Keys to group by

        Returns:
            True if valid, False otherwise
        """
        groups = {}

        for record in records:
            group_key = tuple(record.get(key, '') for key in grouping_keys)

            if group_key not in groups:
                groups[group_key] = 0

            groups[group_key] += 1

        # Check all groups
        for group_key, count in groups.items():
            if count < k:
                logger.warning(
                    f"K-anonymity violation: group {group_key} has {count} records (< {k})"
                )
                return False

        return True

    @staticmethod
    def audit_for_pii(text: str) -> List[str]:
        """
        Audit text for potential PII

        Args:
            text: Text to audit

        Returns:
            List of potential PII matches
        """
        matches = []

        for pattern, label, *flags in PrivacyFilter.PII_PATTERNS:
            regex_flags = flags[0] if flags else 0
            found = re.findall(pattern, text, flags=regex_flags)

            if found:
                matches.extend([f"{label}: {match}" for match in found])

        return matches
