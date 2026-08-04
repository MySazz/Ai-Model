from hybrid_agent.permissions import PermissionPolicy, RiskLevel


def test_observe_is_allowed_without_approval() -> None:
    decision = PermissionPolicy().decide(RiskLevel.OBSERVE)
    assert decision.allowed
    assert not decision.requires_approval


def test_consequential_action_requires_approval() -> None:
    decision = PermissionPolicy().decide(RiskLevel.APPROVAL)
    assert not decision.allowed
    assert decision.requires_approval


def test_prohibited_action_cannot_be_approved() -> None:
    decision = PermissionPolicy().decide(RiskLevel.PROHIBITED, approved=True)
    assert not decision.allowed
