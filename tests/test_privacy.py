from hybrid_agent.privacy import PrivacyClassifier, PrivacyLevel


def test_private_key_is_local_only() -> None:
    assessment = PrivacyClassifier().classify(
        "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----"
    )
    assert assessment.level is PrivacyLevel.LOCAL_ONLY


def test_sensitive_language_requires_cloud_approval() -> None:
    assessment = PrivacyClassifier().classify("Inspect this production configuration")
    assert assessment.level is PrivacyLevel.ASK_BEFORE_CLOUD


def test_ordinary_text_can_use_cloud() -> None:
    assessment = PrivacyClassifier().classify("Explain how a binary tree works")
    assert assessment.level is PrivacyLevel.CLOUD_ALLOWED
