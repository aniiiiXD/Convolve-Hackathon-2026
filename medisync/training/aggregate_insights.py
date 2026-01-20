"""
Aggregate Insights Batch Pipeline

Generates anonymized global medical insights from clinical records:
1. Extracts medical entities from all clinic records
2. Applies K-anonymity and privacy filters
3. Aggregates statistics by condition + treatment
4. Stores in global_medical_insights collection
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from qdrant_client import models

from medisync.core.database import client
from medisync.services.qdrant_ops import COLLECTION_NAME, GLOBAL_INSIGHTS_COLLECTION
from medisync.services.medical_entity_extractor import MedicalEntityExtractor
from medisync.services.embedding import EmbeddingService
from medisync.core.privacy import PrivacyFilter, PrivacyValidator

logger = logging.getLogger(__name__)


class InsightsAggregator:
    """Aggregates clinical records into global insights"""

    def __init__(
        self,
        k_anonymity: int = 20,
        min_clinics: int = 5,
        max_age_days: int = 365
    ):
        """
        Initialize insights aggregator

        Args:
            k_anonymity: Minimum records per insight (K-anonymity parameter)
            min_clinics: Minimum contributing clinics
            max_age_days: Maximum age of records to include
        """
        self.k_anonymity = k_anonymity
        self.min_clinics = min_clinics
        self.max_age_days = max_age_days

        self.entity_extractor = MedicalEntityExtractor()
        self.embedder = EmbeddingService()

    def fetch_clinical_records(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch clinical records from all clinics

        Args:
            limit: Maximum records to fetch (None = all)

        Returns:
            List of clinical records
        """
        logger.info("Fetching clinical records from all clinics...")

        # Calculate date cutoff
        cutoff_timestamp = (
            datetime.utcnow() - timedelta(days=self.max_age_days)
        ).timestamp()

        # Scroll all records (no clinic_id filter for global aggregation)
        records = []
        offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=cutoff_timestamp)
                        ),
                        models.FieldCondition(
                            key="type",
                            match=models.MatchValue(value="note")
                        )
                    ]
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            if not results:
                break

            for point in results:
                record = {
                    'point_id': point.id,
                    'text_content': point.payload.get('text_content', ''),
                    'clinic_id': point.payload.get('clinic_id', ''),
                    'patient_id': point.payload.get('patient_id', ''),
                    'timestamp': point.payload.get('timestamp', 0)
                }
                records.append(record)

            offset = next_offset
            if offset is None or (limit and len(records) >= limit):
                break

        logger.info(f"Fetched {len(records)} clinical records")
        return records[:limit] if limit else records

    def extract_entities_from_records(
        self,
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract medical entities from clinical records

        Args:
            records: List of clinical records

        Returns:
            List of records with extracted entities
        """
        logger.info(f"Extracting entities from {len(records)} records...")

        enriched_records = []

        for i, record in enumerate(records):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(records)} records")

            try:
                entities = self.entity_extractor.extract_entities(
                    record['text_content']
                )

                if entities:
                    enriched_record = {
                        **record,
                        **entities
                    }
                    enriched_records.append(enriched_record)

            except Exception as e:
                logger.warning(f"Error extracting entities from record {i}: {e}")
                continue

        logger.info(
            f"Successfully extracted entities from {len(enriched_records)}/{len(records)} records"
        )

        return enriched_records

    def group_by_condition_treatment(
        self,
        records: List[Dict[str, Any]]
    ) -> Dict[tuple, List[Dict[str, Any]]]:
        """
        Group records by condition and treatment

        Args:
            records: List of enriched records

        Returns:
            Dictionary mapping (condition, treatment) to list of records
        """
        logger.info("Grouping records by condition and treatment...")

        groups = defaultdict(list)

        for record in records:
            condition = record.get('condition', '').lower().strip()
            treatment = record.get('treatment', '').lower().strip()

            if condition and treatment:
                key = (condition, treatment)
                groups[key].append(record)

        logger.info(f"Created {len(groups)} condition-treatment groups")
        return dict(groups)

    def apply_privacy_filters(
        self,
        groups: Dict[tuple, List[Dict[str, Any]]]
    ) -> Dict[tuple, List[Dict[str, Any]]]:
        """
        Apply K-anonymity and privacy filters

        Args:
            groups: Dictionary of grouped records

        Returns:
            Filtered groups satisfying K-anonymity
        """
        logger.info(f"Applying K-anonymity filter (K={self.k_anonymity}, min_clinics={self.min_clinics})...")

        filtered_groups = {}

        for key, records in groups.items():
            # Check K-anonymity
            if len(records) < self.k_anonymity:
                continue

            # Check clinic diversity
            clinic_ids = set(r['clinic_id'] for r in records)
            if len(clinic_ids) < self.min_clinics:
                continue

            # Anonymize records
            anonymized_records = [
                PrivacyFilter.anonymize_record(record)
                for record in records
            ]

            filtered_groups[key] = anonymized_records

        logger.info(
            f"After filtering: {len(filtered_groups)} groups "
            f"(removed {len(groups) - len(filtered_groups)} groups)"
        )

        return filtered_groups

    def aggregate_statistics(
        self,
        groups: Dict[tuple, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Compute aggregated statistics for each group

        Args:
            groups: Filtered groups

        Returns:
            List of insights with statistics
        """
        logger.info("Computing aggregated statistics...")

        insights = []

        for (condition, treatment), records in groups.items():
            try:
                # Aggregate numeric fields
                numeric_fields = ['duration_days']
                categorical_fields = ['outcome', 'severity', 'body_part']

                # Count unique clinics (before anonymization removed clinic_id from records)
                # We need to track this during filtering
                clinic_count = len(set(
                    r.get('clinic_id', '') for r in records if 'clinic_id' in r
                ))

                statistics = PrivacyFilter.aggregate_statistics(
                    records=records,
                    numeric_fields=numeric_fields,
                    categorical_fields=categorical_fields
                )

                # Generate description
                description = self.entity_extractor.generate_insight_description(
                    condition=condition,
                    treatment=treatment,
                    statistics=statistics
                )

                insight = {
                    'insight_id': str(uuid.uuid4()),
                    'insight_type': 'treatment_outcome',
                    'condition': condition,
                    'treatment': treatment,
                    'description': description,
                    'statistics': statistics,
                    'sample_size': statistics['sample_size'],
                    'clinic_count': statistics['clinic_count'],
                    'confidence': min(1.0, statistics['sample_size'] / 100),  # Confidence score
                    'created_at': datetime.utcnow().isoformat()
                }

                insights.append(insight)

            except Exception as e:
                logger.error(f"Error aggregating {condition} + {treatment}: {e}")
                continue

        logger.info(f"Generated {len(insights)} insights")
        return insights

    def store_insights(self, insights: List[Dict[str, Any]]) -> int:
        """
        Store insights in global_medical_insights collection

        Args:
            insights: List of insights to store

        Returns:
            Number of insights stored
        """
        logger.info(f"Storing {len(insights)} insights in Qdrant...")

        points = []

        for insight in insights:
            try:
                # Create embedding for semantic search
                text_to_embed = f"{insight['condition']} {insight['treatment']} {insight['description']}"

                dense_vec = self.embedder.get_dense_embedding(text_to_embed)
                sparse_vec = self.embedder.get_sparse_embedding(text_to_embed)

                point = models.PointStruct(
                    id=insight['insight_id'],
                    vector={
                        "insight_embedding": dense_vec,
                        "sparse_keywords": models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values
                        )
                    },
                    payload=insight
                )

                points.append(point)

            except Exception as e:
                logger.error(f"Error creating point for insight: {e}")
                continue

        # Upsert in batches
        batch_size = 100
        stored_count = 0

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            try:
                client.upsert(
                    collection_name=GLOBAL_INSIGHTS_COLLECTION,
                    points=batch
                )
                stored_count += len(batch)

                logger.info(f"Stored batch {i // batch_size + 1}: {stored_count}/{len(points)} insights")

            except Exception as e:
                logger.error(f"Error storing batch: {e}")

        logger.info(f"✓ Stored {stored_count} insights")
        return stored_count

    def run_aggregation_pipeline(
        self,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete aggregation pipeline

        Args:
            limit: Maximum records to process (None = all)

        Returns:
            Results dictionary
        """
        logger.info("=" * 80)
        logger.info("STARTING INSIGHTS AGGREGATION PIPELINE")
        logger.info("=" * 80)

        results = {
            'status': 'started',
            'timestamp': datetime.utcnow().isoformat(),
            'stages': {}
        }

        try:
            # Stage 1: Fetch records
            records = self.fetch_clinical_records(limit=limit)
            results['stages']['fetch'] = {'record_count': len(records)}

            if len(records) == 0:
                results['status'] = 'no_data'
                return results

            # Stage 2: Extract entities
            enriched_records = self.extract_entities_from_records(records)
            results['stages']['extract'] = {'enriched_count': len(enriched_records)}

            # Stage 3: Group by condition + treatment
            groups = self.group_by_condition_treatment(enriched_records)
            results['stages']['group'] = {'group_count': len(groups)}

            # Stage 4: Apply privacy filters
            filtered_groups = self.apply_privacy_filters(groups)
            results['stages']['filter'] = {'filtered_count': len(filtered_groups)}

            if len(filtered_groups) == 0:
                results['status'] = 'no_valid_insights'
                return results

            # Stage 5: Aggregate statistics
            insights = self.aggregate_statistics(filtered_groups)
            results['stages']['aggregate'] = {'insight_count': len(insights)}

            # Stage 6: Validate privacy
            logger.info("Validating privacy compliance...")
            # Sample validation on first 10 insights
            for insight in insights[:10]:
                description = insight.get('description', '')
                pii_matches = PrivacyFilter.audit_for_pii(description)
                if pii_matches:
                    logger.warning(f"PII detected in insight: {pii_matches}")

            # Stage 7: Store insights
            stored_count = self.store_insights(insights)
            results['stages']['store'] = {'stored_count': stored_count}

            results['status'] = 'completed'
            results['insights_generated'] = len(insights)
            results['insights_stored'] = stored_count

            logger.info("=" * 80)
            logger.info("AGGREGATION PIPELINE COMPLETED")
            logger.info(f"Records processed: {len(records)}")
            logger.info(f"Insights generated: {len(insights)}")
            logger.info(f"Insights stored: {stored_count}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Aggregation pipeline failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)
            return results


def main():
    """CLI entry point for aggregation"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Aggregate global insights")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum records to process"
    )
    parser.add_argument(
        "--k-anonymity",
        type=int,
        default=20,
        help="K-anonymity parameter"
    )
    parser.add_argument(
        "--min-clinics",
        type=int,
        default=5,
        help="Minimum contributing clinics"
    )

    args = parser.parse_args()

    aggregator = InsightsAggregator(
        k_anonymity=args.k_anonymity,
        min_clinics=args.min_clinics
    )

    results = aggregator.run_aggregation_pipeline(limit=args.limit)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
