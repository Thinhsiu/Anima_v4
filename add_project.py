import sys
from utils.project_knowledge import add_project_knowledge, get_project_knowledge

def main():
    """
    Command-line tool to add project information to Anima's knowledge base
    """
    print("===== Add Project to Anima's Knowledge Base =====")
    
    # Get project details from user input
    project_name = input("Project name: ")
    
    project_info = {}
    project_info["description"] = input("Description: ")
    project_info["purpose"] = input("Purpose: ")
    
    # Technologies
    tech_input = input("Technologies (comma-separated): ")
    project_info["technologies"] = [tech.strip() for tech in tech_input.split(",")]
    
    # Features
    features = []
    print("Enter features (one per line, leave empty to finish):")
    while True:
        feature = input("> ")
        if not feature:
            break
        features.append(feature)
    project_info["features"] = features
    
    project_info["status"] = input("Current status: ")
    project_info["github"] = input("GitHub URL (optional): ")
    project_info["website"] = input("Website URL (optional): ")
    project_info["additional_info"] = input("Additional information: ")
    
    # Add the project to knowledge base
    filepath = add_project_knowledge(project_name, project_info)
    
    print(f"\nProject '{project_name}' successfully added to Anima's knowledge base!")
    print(f"Knowledge file: {filepath}")
    
    # Show the added project
    print("\nHere's how Anima will understand your project:")
    projects = get_project_knowledge(project_name)
    if projects:
        print("-" * 50)
        print(projects[0]["content"])
        print("-" * 50)

if __name__ == "__main__":
    main()
