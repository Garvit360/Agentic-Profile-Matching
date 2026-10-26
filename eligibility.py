"""Deterministic eligibility from resume metadata and requirements."""

from __future__ import annotations

from domain_models import (
    AssessmentStatus,
    CriterionAssessment,
    CriterionKind,
    EligibilityStatus,
    RequirementSet,
)


def assess_eligibility(
    resume_metadata: dict,
    requirements: RequirementSet,
) -> tuple[EligibilityStatus, list[CriterionAssessment]]:
    assessments: list[CriterionAssessment] = []
    has_not_met = False
    has_unknown = False

    min_years = requirements.minimum_experience_years
    if min_years is not None:
        exp = resume_metadata.get("experience_years")
        if exp is None or exp < 0:
            assessments.append(
                CriterionAssessment(
                    criterion_id="experience-minimum",
                    status=AssessmentStatus.UNKNOWN,
                    evidence_excerpt=None,
                )
            )
            has_unknown = True
        elif exp < min_years:
            assessments.append(
                CriterionAssessment(
                    criterion_id="experience-minimum",
                    status=AssessmentStatus.NOT_MET,
                    evidence_excerpt=f"Parsed experience: {exp} years",
                )
            )
            has_not_met = True
        else:
            assessments.append(
                CriterionAssessment(
                    criterion_id="experience-minimum",
                    status=AssessmentStatus.MET,
                    evidence_excerpt=f"Parsed experience: {exp} years",
                )
            )

    meta_skills_str = resume_metadata.get("skills") or ""
    meta_skills = [x.strip().lower() for x in meta_skills_str.split(",") if x.strip()]

    for crit in requirements.must_have:
        if crit.kind == CriterionKind.EXPERIENCE:
            continue
        label = crit.label.lower()
        if not label:
            continue
        if any(label in ms or ms in label for ms in meta_skills):
            assessments.append(
                CriterionAssessment(criterion_id=crit.id, status=AssessmentStatus.MET, evidence_excerpt=crit.label)
            )
        elif not meta_skills_str:
            assessments.append(
                CriterionAssessment(criterion_id=crit.id, status=AssessmentStatus.UNKNOWN, evidence_excerpt=None)
            )
            has_unknown = True
        else:
            matched = any(label.split()[0] in ms for ms in meta_skills if label.split())
            if matched:
                assessments.append(
                    CriterionAssessment(criterion_id=crit.id, status=AssessmentStatus.MET, evidence_excerpt=crit.label)
                )
            else:
                assessments.append(
                    CriterionAssessment(
                        criterion_id=crit.id,
                        status=AssessmentStatus.UNKNOWN,
                        evidence_excerpt="No verified skill mention in indexed metadata",
                    )
                )
                has_unknown = True

    if has_not_met:
        return EligibilityStatus.INELIGIBLE, assessments
    if has_unknown:
        return EligibilityStatus.PENDING_REVIEW, assessments
    if requirements.must_have or min_years is not None:
        return EligibilityStatus.ELIGIBLE, assessments
    return EligibilityStatus.ELIGIBLE, assessments


def combined_score(initial: int, deep_fit: int) -> int:
    return int((initial * 0.6) + (deep_fit * 0.4))


def recommendation_from_scores(
    eligibility: EligibilityStatus,
    analysis_complete: bool,
    combined: int,
) -> str:
    from domain_models import FinalRecommendation

    if eligibility != EligibilityStatus.ELIGIBLE or not analysis_complete:
        return FinalRecommendation.PENDING.value
    if combined >= 80:
        return FinalRecommendation.HIRE.value
    if combined >= 60:
        return FinalRecommendation.NO_HIRE_FOR_NOW.value
    return FinalRecommendation.NO_HIRE.value
