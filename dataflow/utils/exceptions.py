class NoSuchExperimentException(Exception):
    def __init__(self, project_name: str, step_name: str):
        self.project_name = project_name
        self.step_name = step_name
        self.servename = "server"

    def __str__(self):
        err_lines = [
            f"{self.servename} knows no project {self.project_name} with step {self.step_name}",
            "To create it, run the following",
            "> project_main_dir = '/some/root/path/on/your/storage",
            f"> DataTracker.create_dataset_experiment({self.project_name}, {self.step_name}, project_main_dir)",
        ]
        return "\n".join(err_lines)