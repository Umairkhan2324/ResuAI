from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class PersonalInfo(BaseModel):
    """Personal information model"""
    name: str = Field(default="")
    email: Optional[str] = Field(default="")
    phone: Optional[str] = Field(default="")
    location: Optional[str] = Field(default="")

class Experience(BaseModel):
    """Work experience model"""
    title: str = Field(default="")
    company: str = Field(default="")
    duration: str = Field(default="")
    description: List[str] = Field(default_factory=list)

class Education(BaseModel):
    """Education model"""
    degree: str = Field(default="")
    institution: str = Field(default="")
    year: str = Field(default="")
    details: Optional[str] = Field(default="")

class ProfessionalInfo(BaseModel):
    """Professional information model"""
    experiences: List[Experience] = Field(default_factory=lambda: [Experience()])
    education: List[Education] = Field(default_factory=lambda: [Education()])
    skills: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)

class JobAnalysis(BaseModel):
    """Job market analysis model"""
    field: str = Field(default="")
    required_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    industry_trends: List[str] = Field(default_factory=list)
    career_paths: List[str] = Field(default_factory=list)
    ats_keywords: List[str] = Field(default_factory=list)

class ResumeData(BaseModel):
    """Complete resume data model"""
    job_field: str = Field(default="")
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    professional_info: ProfessionalInfo = Field(default_factory=ProfessionalInfo)
    job_analysis: JobAnalysis = Field(default_factory=JobAnalysis)
    last_updated: datetime = Field(default_factory=datetime.now)

class ConversationMessage(BaseModel):
    """Conversation message model"""
    role: str
    message: str
    time: datetime = Field(default_factory=datetime.now)

class AgentResponse(BaseModel):
    """Standardized agent response model"""
    type: str
    content: Dict
    analysis: Optional[str] = Field(default="")
    next_action: Optional[str] = Field(default="") 