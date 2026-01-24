from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Violation(BaseModel):
    article: str
    code: str
    context: Dict[str, Any]


class SuggestReq(BaseModel):
    project_id: Optional[str] = None
    xml: Optional[str] = None
    violations: List[Violation]
    attachments: Optional[List[Dict[str, Any]]] = None


class Suggestion(BaseModel):
    violation_ref: int
    fix: str
    nec_refs: List[str]
    confidence: float
    rationale: Optional[str] = None
    priority: float


class SuggestResp(BaseModel):
    suggestions: List[Suggestion]
    notes: Optional[List[str]] = []
