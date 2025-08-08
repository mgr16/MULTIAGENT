from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class RouterOutput(BaseModel):
    domain: Literal["finance","web","vision","code","data","general"] = "general"
    confidence: float = 0.5
    suggested_agents: List[str] = Field(default_factory=list)

class PlanStep(BaseModel):
    name: str
    agents: List[str]
    requires: List[str] = Field(default_factory=list)
    parallel_group: Optional[str] = None

class PlanOutput(BaseModel):
    steps: List[PlanStep]
    stop_condition: str = "final_answer"

class VisionStruct(BaseModel):
    contains_chart: bool
    contains_text: bool
    any_numbers: bool
    chart_type: Optional[str] = None
    main_info: Optional[str] = None

class RAGPassage(BaseModel):
    source: str
    span: Optional[str] = None
    text: str

class RAGOutput(BaseModel):
    passages: List[RAGPassage]

class CriticIssue(BaseModel):
    kind: Literal["missing_citation","contradiction","math_error","format_error","other"] = "other"
    detail: str

class CriticVerdict(BaseModel):
    confidence: float = 0.6
    conflicts: int = 0
    issues: List[CriticIssue] = Field(default_factory=list)

class SummaryOutput(BaseModel):
    final_answer: str
    citations: List[str] = Field(default_factory=list)
