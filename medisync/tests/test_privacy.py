"""
Privacy Compliance Tests

Tests K-anonymity, PII removal, and HIPAA compliance features.
"""

import pytest
from medisync.core.privacy import PrivacyFilter, PrivacyValidator


class TestPIIRemoval:
    """Test PII detection and removal"""

    def test_ssn_removal(self):
        """Test SSN pattern removal"""
        text = "Patient SSN: 123-45-6789"
        sanitized = PrivacyFilter.remove_pii(text)

        assert "123-45-6789" not in sanitized
        assert "[SSN]" in sanitized

    def test_phone_removal(self):
        """Test phone number removal"""
        text = "Call 555-123-4567 for appointment"
        sanitized = PrivacyFilter.remove_pii(text)

        assert "555-123-4567" not in sanitized
        assert "[PHONE]" in sanitized

    def test_email_removal(self):
        """Test email address removal"""
        text = "Contact: patient@example.com"
        sanitized = PrivacyFilter.remove_pii(text)

        assert "patient@example.com" not in sanitized
        assert "[EMAIL]" in sanitized

    def test_id_removal(self):
        """Test patient/doctor ID removal"""
        text = "Patient ID: P12345"
        sanitized = PrivacyFilter.remove_pii(text)

        assert "P12345" not in sanitized
        assert "[ID]" in sanitized

    def test_address_removal(self):
        """Test address removal"""
        text = "Lives at 123 Main Street"
        sanitized = PrivacyFilter.remove_pii(text)

        assert "Main Street" not in sanitized or "[ADDRESS]" in sanitized

    def test_multiple_pii_removal(self):
        """Test removal of multiple PII types"""
        text = "John Doe (SSN: 123-45-6789) at 555-123-4567, email: john@example.com"
        sanitized = PrivacyFilter.remove_pii(text)

        assert "123-45-6789" not in sanitized
        assert "555-123-4567" not in sanitized
        assert "john@example.com" not in sanitized


class TestGeneralization:
    """Test generalization hierarchies"""

    def test_age_generalization(self):
        """Test age bracket generalization"""
        assert PrivacyFilter.generalize_age(5) == "0-18"
        assert PrivacyFilter.generalize_age(25) == "18-30"
        assert PrivacyFilter.generalize_age(35) == "30-40"
        assert PrivacyFilter.generalize_age(45) == "40-50"
        assert PrivacyFilter.generalize_age(55) == "50-60"
        assert PrivacyFilter.generalize_age(65) == "60-70"
        assert PrivacyFilter.generalize_age(80) == "70+"

    def test_body_part_generalization(self):
        """Test body part generalization"""
        assert PrivacyFilter.generalize_body_part("right index finger") == "finger"
        assert PrivacyFilter.generalize_body_part("left knee") == "knee"
        assert PrivacyFilter.generalize_body_part("upper arm") == "arm"
        assert PrivacyFilter.generalize_body_part("forearm") == "arm"


class TestKAnonymity:
    """Test K-anonymity enforcement"""

    def test_basic_k_anonymity(self):
        """Test basic K-anonymity filtering"""
        records = [
            {"condition": "fracture", "treatment": "cast", "clinic_id": f"clinic_{i}"}
            for i in range(25)
        ]

        # Should pass with K=20
        filtered = PrivacyFilter.apply_k_anonymity(
            records=records,
            k=20,
            min_clinics=1,
            grouping_keys=['condition', 'treatment']
        )

        assert len(filtered) == 25

    def test_k_anonymity_rejection(self):
        """Test that groups < K are filtered out"""
        records = [
            {"condition": "fracture", "treatment": "cast", "clinic_id": f"clinic_{i}"}
            for i in range(15)
        ]

        # Should fail with K=20
        filtered = PrivacyFilter.apply_k_anonymity(
            records=records,
            k=20,
            min_clinics=1,
            grouping_keys=['condition', 'treatment']
        )

        assert len(filtered) == 0

    def test_clinic_diversity_requirement(self):
        """Test minimum clinic count requirement"""
        # 25 records but only 3 clinics
        records = [
            {"condition": "fracture", "treatment": "cast", "clinic_id": f"clinic_{i % 3}"}
            for i in range(25)
        ]

        # Should fail with min_clinics=5
        filtered = PrivacyFilter.apply_k_anonymity(
            records=records,
            k=20,
            min_clinics=5,
            grouping_keys=['condition', 'treatment']
        )

        assert len(filtered) == 0

    def test_multiple_groups(self):
        """Test K-anonymity with multiple condition-treatment groups"""
        records = []

        # Group 1: fracture + cast (25 records, passes)
        for i in range(25):
            records.append({
                "condition": "fracture",
                "treatment": "cast",
                "clinic_id": f"clinic_{i % 10}"
            })

        # Group 2: diabetes + metformin (15 records, fails)
        for i in range(15):
            records.append({
                "condition": "diabetes",
                "treatment": "metformin",
                "clinic_id": f"clinic_{i % 10}"
            })

        filtered = PrivacyFilter.apply_k_anonymity(
            records=records,
            k=20,
            min_clinics=5,
            grouping_keys=['condition', 'treatment']
        )

        # Should only have fracture+cast group
        assert len(filtered) == 25
        assert all(r['condition'] == 'fracture' for r in filtered)


class TestOutlierSuppression:
    """Test outlier suppression"""

    def test_outlier_removal(self):
        """Test that outliers are removed"""
        values = list(range(1, 101))  # 1 to 100

        filtered = PrivacyFilter.suppress_outliers(values, percentile=5.0)

        # Should remove bottom 5% (1-5) and top 5% (96-100)
        assert 1 not in filtered
        assert 5 not in filtered
        assert 96 not in filtered
        assert 100 not in filtered

        # Middle values should remain
        assert 50 in filtered

    def test_empty_list(self):
        """Test handling of empty list"""
        filtered = PrivacyFilter.suppress_outliers([])
        assert filtered == []


class TestRecordAnonymization:
    """Test record anonymization"""

    def test_anonymize_record(self):
        """Test full record anonymization"""
        record = {
            "patient_id": "P-123",
            "clinic_id": "C-456",
            "doctor_id": "D-789",
            "condition": "fracture",
            "treatment": "cast",
            "age": 45,
            "body_part": "right index finger",
            "text": "Patient 123-45-6789 called 555-123-4567"
        }

        anonymized = PrivacyFilter.anonymize_record(record)

        # Should remove identifiers
        assert "patient_id" not in anonymized
        assert "clinic_id" not in anonymized
        assert "doctor_id" not in anonymized

        # Should keep medical fields
        assert anonymized['condition'] == "fracture"
        assert anonymized['treatment'] == "cast"

        # Should generalize age
        assert anonymized['age'] == "40-50"

        # Should generalize body part
        assert anonymized['body_part'] == "finger"

        # Should remove PII from text
        assert "123-45-6789" not in anonymized['text']
        assert "555-123-4567" not in anonymized['text']


class TestStatisticsAggregation:
    """Test aggregated statistics"""

    def test_numeric_aggregation(self):
        """Test numeric field aggregation"""
        records = [
            {"duration_days": i, "clinic_id": f"c{i % 5}"}
            for i in range(10, 50)
        ]

        stats = PrivacyFilter.aggregate_statistics(
            records=records,
            numeric_fields=['duration_days'],
            categorical_fields=[]
        )

        assert 'sample_size' in stats
        assert 'clinic_count' in stats
        assert 'duration_days_median' in stats
        assert 'duration_days_mean' in stats

        assert stats['sample_size'] == len(records)
        assert stats['clinic_count'] == 5

    def test_categorical_aggregation(self):
        """Test categorical field aggregation"""
        records = [
            {"outcome": "healed", "clinic_id": "c1"} for _ in range(17)
        ] + [
            {"outcome": "complications", "clinic_id": "c2"} for _ in range(3)
        ]

        stats = PrivacyFilter.aggregate_statistics(
            records=records,
            numeric_fields=[],
            categorical_fields=['outcome']
        )

        assert 'outcome_distribution' in stats
        assert stats['outcome_distribution']['healed'] == 0.85  # 17/20
        assert stats['outcome_distribution']['complications'] == 0.15  # 3/20


class TestPrivacyValidator:
    """Test privacy validation"""

    def test_k_anonymity_validation(self):
        """Test K-anonymity validation"""
        # Valid: 25 records in one group
        records = [
            {"condition": "fracture", "treatment": "cast"}
            for _ in range(25)
        ]

        assert PrivacyValidator.validate_k_anonymity(
            records=records,
            k=20,
            grouping_keys=['condition', 'treatment']
        ) is True

        # Invalid: 15 records in one group
        records = [
            {"condition": "fracture", "treatment": "cast"}
            for _ in range(15)
        ]

        assert PrivacyValidator.validate_k_anonymity(
            records=records,
            k=20,
            grouping_keys=['condition', 'treatment']
        ) is False

    def test_pii_audit(self):
        """Test PII auditing"""
        text_with_pii = "SSN: 123-45-6789, Phone: 555-123-4567"
        matches = PrivacyValidator.audit_for_pii(text_with_pii)

        assert len(matches) >= 2
        assert any("SSN" in match for match in matches)
        assert any("PHONE" in match for match in matches)

        text_without_pii = "Patient has a fracture"
        matches = PrivacyValidator.audit_for_pii(text_without_pii)

        assert len(matches) == 0


class TestHIPAACompliance:
    """Test HIPAA compliance features"""

    def test_no_phi_in_insights(self):
        """Test that no PHI appears in aggregated insights"""
        records = [
            {
                "patient_id": f"P-{i}",
                "clinic_id": f"C-{i % 5}",
                "condition": "fracture",
                "treatment": "cast",
                "age": 40 + i,
                "text": f"Patient SSN 123-45-67{i:02d}"
            }
            for i in range(25)
        ]

        # Anonymize all records
        anonymized = [
            PrivacyFilter.anonymize_record(r, remove_keys={'patient_id', 'clinic_id'})
            for r in records
        ]

        # Check no PHI
        for record in anonymized:
            assert "patient_id" not in record
            assert "clinic_id" not in record
            assert "SSN" not in str(record)

            # Verify PII was removed from text
            if 'text' in record:
                pii_matches = PrivacyValidator.audit_for_pii(record['text'])
                assert len(pii_matches) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
