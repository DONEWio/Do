
from typing import Literal


def test_pydantic_to_simple_schema():
    from pydantic import BaseModel, Field
    from donew.utils import pydantic_model_to_simple_schema

    class Occupation(BaseModel):
        name: str = Field(description="The name of the occupation", default="unknown")
        description: str = Field(description="The description of the occupation")

    class Hobby(BaseModel):
        name: str = Field(description="The name of the hobby", required=True)
        years: int|str = Field(description="The number of years the hobby has been practiced")

    class Persona(BaseModel):
        name: str = Field(description="The name of the person", required=True)
        age: int = Field(description="The age of the person")
        gender: Literal["male", "female", "other"] = Field(description="The gender of the person")
        occupation: Occupation
        interests: list[str] = Field(description="The interests of the person", required=True, default=[])
        hobbies: list[Hobby]
    
    format_json = pydantic_model_to_simple_schema(Persona)
    assert format_json == {
        "name": "<string> The name of the person [REQUIRED]",
        "age": "<integer> The age of the person",
        "gender": "<string> The gender of the person [CHOICES: male|female|other]",
        "occupation": {
            "name": "<string> The name of the occupation [DEFAULT: unknown]",
            "description": "<string> The description of the occupation"
        },
        "interests": "<array[string]> The interests of the person [REQUIRED] [DEFAULT: []]",
        "hobbies": [
            {
                "name": "<string> The name of the hobby [REQUIRED]",
                "years": "<integer|string> The number of years the hobby has been practiced"
            }
        ]
    }
