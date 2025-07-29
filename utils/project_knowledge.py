import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import knowledge_manager
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.knowledge_manager import save_knowledge, get_knowledge

def add_project_knowledge(project_name, project_info):
    """
    Add knowledge about a project to Anima's knowledge base.
    
    Parameters:
    - project_name: Name of the project
    - project_info: Dictionary containing project details
    
    Project info should include:
    - description: Brief description of the project
    - purpose: What problem the project solves
    - technologies: List of technologies used
    - features: Key features of the project
    - status: Current status (e.g., "in development", "completed")
    - github: GitHub repository URL (optional)
    - website: Project website URL (optional)
    - additional_info: Any other relevant information
    
    Returns:
    - Path to the created knowledge file
    """
    # Format the topic to clearly identify it as a project
    topic = f"PROJECT: {project_name}"
    
    # Convert project_info to a well-formatted string
    content = f"# {project_name}\n\n"
    
    if "description" in project_info:
        content += f"## Description\n{project_info['description']}\n\n"
    
    if "purpose" in project_info:
        content += f"## Purpose\n{project_info['purpose']}\n\n"
    
    if "technologies" in project_info:
        content += "## Technologies\n"
        if isinstance(project_info["technologies"], list):
            for tech in project_info["technologies"]:
                content += f"- {tech}\n"
        else:
            content += f"{project_info['technologies']}\n"
        content += "\n"
    
    if "features" in project_info:
        content += "## Features\n"
        if isinstance(project_info["features"], list):
            for feature in project_info["features"]:
                content += f"- {feature}\n"
        else:
            content += f"{project_info['features']}\n"
        content += "\n"
    
    if "status" in project_info:
        content += f"## Status\n{project_info['status']}\n\n"
    
    if "github" in project_info and project_info["github"]:
        content += f"## GitHub\n{project_info['github']}\n\n"
    
    if "website" in project_info and project_info["website"]:
        content += f"## Website\n{project_info['website']}\n\n"
    
    if "additional_info" in project_info:
        content += f"## Additional Information\n{project_info['additional_info']}\n\n"
    
    # Save the knowledge
    return save_knowledge(topic, content)

def get_project_knowledge(project_name=None):
    """
    Retrieve knowledge about projects.
    
    Parameters:
    - project_name: Optional name of a specific project to retrieve
    
    Returns:
    - List of project knowledge items
    """
    search_term = "PROJECT:" if project_name is None else f"PROJECT: {project_name}"
    return get_knowledge(search_term)

def update_project_status(project_name, new_status):
    """
    Update the status of an existing project.
    
    Parameters:
    - project_name: Name of the project to update
    - new_status: New status of the project
    
    Returns:
    - True if successful, False otherwise
    """
    # Get existing project knowledge
    projects = get_project_knowledge(project_name)
    
    if not projects:
        return False
    
    # Get the most recent project entry
    project = projects[0]
    
    # Parse content and update status
    lines = project["content"].split("\n")
    new_content = []
    status_updated = False
    
    for i, line in enumerate(lines):
        if line == "## Status":
            new_content.append(line)
            new_content.append(new_status)
            status_updated = True
            # Skip the next line (old status)
            continue
        elif status_updated and i > 0 and lines[i-1] == "## Status":
            continue
        else:
            new_content.append(line)
    
    # If status section wasn't found, add it
    if not status_updated:
        new_content.append("## Status")
        new_content.append(new_status)
    
    # Save updated content
    updated_content = "\n".join(new_content)
    save_knowledge(project["topic"], updated_content)
    return True

# Example usage
if __name__ == "__main__":
    # Example: Adding a new project
    project_info = {
        "description": "A web application for tracking personal finances",
        "purpose": "Help users manage their spending and saving habits",
        "technologies": ["Python", "Flask", "React", "PostgreSQL"],
        "features": [
            "Expense tracking",
            "Budget planning",
            "Financial reports",
            "Goal setting"
        ],
        "status": "In development (70% complete)",
        "github": "https://github.com/username/finance-tracker",
        "additional_info": "Expected completion date: August 2025"
    }
    
    add_project_knowledge("Finance Tracker", project_info)
