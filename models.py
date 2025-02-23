from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class PersonalInfo(BaseModel):
    """Personal information model"""
    name: str = Field(default="", description="Full name")
    email: Optional[str] = Field(default="", description="Email address")
    phone: Optional[str] = Field(default="", description="Contact number")
    location: Optional[str] = Field(default="", description="Current location")

class Experience(BaseModel):
    """Work experience model"""
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    duration: str = Field(..., description="Employment duration")
    description: List[str] = Field(default_list=[], description="Key responsibilities and achievements")

class Education(BaseModel):
    """Education model"""
    degree: str = Field(..., description="Degree name")
    institution: str = Field(..., description="Institution name")
    year: str = Field(..., description="Completion year")
    details: Optional[str] = Field(default="", description="Additional details")

class ProfessionalInfo(BaseModel):
    """Professional information model"""
    experiences: List[Experience] = Field(default_list=[], description="Work experiences")
    education: List[Education] = Field(default_list=[], description="Educational background")
    skills: List[str] = Field(default_list=[], description="Technical and soft skills")
    achievements: List[str] = Field(default_list=[], description="Notable achievements")

class JobAnalysis(BaseModel):
    """Job market analysis model"""
    field: str = Field(..., description="Job field/title")
    required_skills: List[str] = Field(..., description="Required technical skills")
    soft_skills: List[str] = Field(..., description="Required soft skills")
    qualifications: List[str] = Field(..., description="Required qualifications")
    industry_trends: List[str] = Field(..., description="Current industry trends")
    career_paths: List[str] = Field(..., description="Possible career progression paths")
    ats_keywords: List[str] = Field(..., description="Important ATS keywords")

class ResumeData(BaseModel):
    """Complete resume data model"""
    job_field: str = Field(..., description="Target job field")
    personal_info: PersonalInfo = Field(..., description="Personal information")
    professional_info: ProfessionalInfo = Field(..., description="Professional information")
    job_analysis: JobAnalysis = Field(..., description="Job market analysis")
    last_updated: datetime = Field(default_factory=datetime.now)

class ConversationMessage(BaseModel):
    """Conversation message model"""
    role: str = Field(..., description="Message sender (user/assistant)")
    message: str = Field(..., description="Message content")
    time: datetime = Field(default_factory=datetime.now)

class AgentResponse(BaseModel):
    """Standardized agent response model"""
    type: str = Field(..., description="Response type (message/resume)")
    content: Dict = Field(..., description="Response content")
    analysis: Optional[str] = Field(default="", description="Agent's analysis")
    next_action: Optional[str] = Field(default="", description="Suggested next action") 