from typing import List, Optional

from pydantic import BaseModel

from .chat_completion_request import ChatCompletionRequest
from .message import Message


class SceneObject(BaseModel):
    """ SceneObject represents an object in a scene. 

    Attributes:
        label (str): The name of the object.
        id (str): The instance ID of the object.
        attributes (List[str]): A list of attributes of the object.
    """
    label: str
    id: Optional[str] = None
    attributes: Optional[List[str]] = None

    def __repr__(self) -> str:
        repr = f"obj-{self.label}"
        if self.id is not None:
            repr = f"{repr}-{self.id}"
        if self.attributes is not None:
            repr = f"{repr}:[{', '.join(self.attributes)}]"
        return repr


class SceneRelation(BaseModel):
    """ SceneRelation represents a relationship between two objects in a scene. 

    Attributes:
        subject (str): The subject of the relationship.
        predicate (str): The predicate of the relationship.
        object (str): The object of the relationship.
    """
    subject: str
    predicate: str
    object: str
    id: Optional[str] = None

    def __repr__(self) -> str:
        repr = f"rel"
        if self.id is not None:
            repr = f"{repr}-{self.id}"
        return f"{repr}:({self.subject}, {self.predicate}, {self.object})"


class SceneGraph(BaseModel):
    """ SceneGraph represents a scene graph that consists of objects and relationships. 

    Attributes:
        objects (List[SceneObject]): A list of objects in the scene.
        relations (List[SceneRelation]): A list of relationships between objects in the scene.
    """
    objects: List[SceneObject]
    relations: List[SceneRelation]

    def __repr__(self) -> str:
        objects = self.objects or []
        relations = self.relations or []

        repr = "{"
        repr += "; ".join([str(obj) for obj in objects])
        repr += "; "
        repr += "; ".join([str(rel) for rel in relations])
        repr += "}"

        return repr


class InstructiosRequest(ChatCompletionRequest):
    """ InstructiosRequest represents a request for generating instructions from a scene graph. 

    Attributes:
        scene_graph (SceneGraph): The scene graph to generate instructions from.
        task (str): The task to generate instructions for.
    """
    scene_graph: SceneGraph
    task: str
    messages: List[Message] = None