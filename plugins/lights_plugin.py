import logging
from semantic_kernel.functions import kernel_function


logging.getLogger("kernel").setLevel(logging.DEBUG)


class LightsPlugin:
    lights = [
        {"id": 1, "name": "Table Lamp", "is_on": False},
        {"id": 2, "name": "Porch light", "is_on": False},
        {"id": 3, "name": "Chandelier", "is_on": True},
    ]

    @kernel_function(
        name="get_lights",
        description="Gets a list of lights and their current state",
    )
    def get_state(
        self,
    ) -> str:
        
        print("Getting the state of all lights")
        
        """Gets a list of lights and their current state."""
        return self.lights

    @kernel_function(
        name="change_state",
        description="Changes the state of the light",
    )
    def change_state(
        self,
        id: int,
        is_on: bool,
    ) -> str:
        
        print(f"Changing state of light with id '{id}' to '{is_on}'")

        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = is_on
                return light
        return None