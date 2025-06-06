import logging
from semantic_kernel.functions import kernel_function


logging.getLogger("kernel").setLevel(logging.DEBUG)


class AutomationPlugin:

    @kernel_function(
        name="request_automation",
        description="Requests automation on behalf of a person for a specific task or process"
    )
    def request_automation(
        self,
        first_name: str,
        last_name: str,
        email_address: str,
        task_or_process_name: str
    ) -> str:
        
        print(f"Automation requested for {first_name} {last_name} with email address {email_address} for task/process named '{task_or_process_name}'")
        
        """Gets a list of lights and their current state."""
        return None