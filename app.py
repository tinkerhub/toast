from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, Body
from facenet_pytorch import MTCNN, InceptionResnetV1
from pydantic import BaseModel

from utils import get_faces

app = FastAPI()

# Initialize the MTCNN face detection model and InceptionResnetV1 face recognition model
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize a global variable to store the image embeddings
image_embeddings = []
user_ids = []


class UserResponse(BaseModel):
    user_id: str


@app.post("/image_to_user", response_model=List[str])
async def image_to_user(image: bytes = File(...)):
    # Extract faces from the image
    faces = get_faces(image)

    ids = []

    for face in faces:
        # Convert the face data from a numpy array to a PyTorch tensor
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float()

        # Compute the embedding for the face using the InceptionResnetV1 model
        embedding = resnet(face_tensor.unsqueeze(0)).detach().numpy().reshape(-1)

        # Compute cosine similarity between the face embedding and the stored embeddings
        similarities = np.dot(image_embeddings, embedding.T) / (
                    np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(embedding))

        # Find the index of the most similar face
        most_similar_index = np.argmax(similarities)

        # Your code to get the user id associated with the most similar face
        user_id = user_ids[most_similar_index]

        ids.append({"user_id": user_id})

    return user_ids


@app.post("/user", response_model=UserResponse)
async def create_user(image: bytes = File(...), user_id: str = Body(...)):
    # Extract faces from the image
    faces = get_faces(image)

    for face in faces:
        # Convert the face data from a numpy array to a PyTorch tensor
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float()

        # Compute the embedding for the face using the InceptionResnetV1 model
        embedding = resnet(face_tensor.unsqueeze(0)).detach().squeeze().numpy()

        # Store the image embedding in the global variable
        global image_embeddings, user_ids
        image_embeddings.append(embedding)
        user_ids.append(user_id)

    # Your code to create the user profile
    return {"user_id": user_id}


@app.delete("/user/{user_id}", response_model=UserResponse)
async def delete_user(user_id: str):
    # Your code to delete the user profile and associated image embedding(s)
    return {"user_id": user_id}
