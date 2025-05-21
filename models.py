from pydantic import BaseModel

class RepoLink(BaseModel):
    """Model for GitHub repository URL input."""
    repo_url: str
    
class UserMessage(BaseModel):
    """Model for user questions to the code assistant."""
    message: str 