"""Convert LLM requirement specs into canonical RequirementSet and refinements."""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from domain_models import (
    Criterion,
    CriterionKind,
    RequirementSet,
    criterion_id_from_label,
    requirement_set_from_spec,
)


class RequirementSpecPayload(BaseModel):
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    years_experience: Optional[int] = None
    role_context: str = ""


class RefinementResult(BaseModel):
    requirements: RequirementSet
    applied: bool
    clarification: Optional[str] = None


def spec_to_requirements(spec: RequirementSpecPayload) -> RequirementSet:
    return requirement_set_from_spec(
        must_have=spec.must_have,
        nice_to_have=spec.nice_to_have,
        years_experience=spec.years_experience,
        role_context=spec.role_context,
    )


def merge_initial_jd_requirements(jd: str, spec: RequirementSpecPayload) -> RequirementSet:
    base = spec_to_requirements(spec)
    if base.minimum_experience_years is None:
        years = _parse_years_from_text(jd)
        if years is not None:
            return base.model_copy(update={"minimum_experience_years": years})
    return base


def apply_refinement(current: RequirementSet, message: str) -> RefinementResult:
    msg = message.lower().strip()
    updated = current.model_copy(deep=True)
    changed = False

    years_match = re.search(r"(\d+)\s*\+?\s*years?", msg, re.I)
    if years_match and any(k in msg for k in ("require", "years", "experience", "actually")):
        updated.minimum_experience_years = int(years_match.group(1))
        changed = True

    optional_match = re.search(r"(\w[\w\s/+.-]{1,40})\s+is\s+optional", msg, re.I)
    if optional_match:
        label = optional_match.group(1).strip()
        updated = _move_to_nice_to_have(updated, label)
        changed = True

    remove_match = re.search(r"remove\s+(\w[\w\s/+.-]{1,40})", msg, re.I)
    if remove_match:
        label = remove_match.group(1).strip()
        updated = _remove_criterion(updated, label)
        changed = True

    add_match = re.search(r"(?:require|add)\s+(.+)", msg)
    if add_match:
        for label in re.split(r"\s+and\s+|,", add_match.group(1).rstrip(".")):
            label = label.strip()
            if not re.fullmatch(r"\d+\+?\s*years?(?:\s+experience)?", label):
                updated = _add_must_have(updated, label)
                changed = True

    if not changed and any(k in msg for k in ("actually", "refine", "update", "change")):
        if years_match or "aws" in msg or "optional" in msg or "remove" in msg:
            pass
        else:
            return RefinementResult(
                requirements=current,
                applied=False,
                clarification=(
                    "I need a clearer refinement. Examples: 'require 5+ years', "
                    "'AWS is optional', or 'remove AWS'."
                ),
            )

    if not changed:
        return RefinementResult(requirements=current, applied=False, clarification="Please specify the requirement to add, remove, or change.")

    return RefinementResult(requirements=updated, applied=True, clarification=None)


def _parse_years_from_text(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*\+?\s*years?", text, re.I)
    return int(m.group(1)) if m else None


def _move_to_nice_to_have(req: RequirementSet, label: str) -> RequirementSet:
    label_l = label.lower()
    remaining: list[Criterion] = []
    moved: Optional[Criterion] = None
    for c in req.must_have:
        if label_l in c.label.lower():
            moved = c.model_copy(update={"kind": CriterionKind.SKILL})
        else:
            remaining.append(c)
    nice = list(req.nice_to_have)
    if moved:
        nice.append(moved)
    else:
        nice.append(Criterion(id=criterion_id_from_label(label), label=label, kind=CriterionKind.SKILL))
    return req.model_copy(update={"must_have": remaining, "nice_to_have": nice})


def _remove_criterion(req: RequirementSet, label: str) -> RequirementSet:
    label_l = label.lower()
    must = [c for c in req.must_have if label_l not in c.label.lower()]
    nice = [c for c in req.nice_to_have if label_l not in c.label.lower()]
    return req.model_copy(update={"must_have": must, "nice_to_have": nice})


def _add_must_have(req: RequirementSet, label: str) -> RequirementSet:
    if any(label.lower() in c.label.lower() for c in req.must_have):
        return req
    crit = Criterion(id=criterion_id_from_label(label), label=label, kind=CriterionKind.SKILL)
    return req.model_copy(update={"must_have": [*req.must_have, crit]})
