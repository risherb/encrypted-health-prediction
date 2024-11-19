"""Server that will listen for GET and POST requests from the client."""

import time
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from utils import DEPLOYMENT_DIR, SERVER_DIR  # pylint: disable=no-name-in-module

from concrete.ml.deployment import FHEModelServer

# Load the FHE server
FHE_SERVER = FHEModelServer(DEPLOYMENT_DIR)

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route
@app.get("/")
def root():
    """
    Root endpoint of the health prediction API.

    Returns:
        dict: The welcome message.
    """
    return {"message": "Welcome to your disease prediction with FHE!"}


@app.post("/send_input")
def send_input(
    user_id: str = Form(),
    files: List[UploadFile] = File(),
):
    """Send the inputs to the server."""

    print("\nSend the data to the server ............\n")

    # Receive the Client's files (Evaluation key + Encrypted symptoms)
    evaluation_key_path = SERVER_DIR / f"{user_id}_valuation_key"
    encrypted_input_path = SERVER_DIR / f"{user_id}_encrypted_input"

    # Save the files using the above paths
    with encrypted_input_path.open("wb") as encrypted_input, evaluation_key_path.open(
        "wb"
    ) as evaluation_key:
        encrypted_input.write(files[0].file.read())
        evaluation_key.write(files[1].file.read())


@app.post("/run_fhe")
def run_fhe(
    user_id: str = Form(),
):
    """Inference in FHE."""

    print("\nRun in FHE in the server ............\n")
    evaluation_key_path = SERVER_DIR / f"{user_id}_valuation_key"
    encrypted_input_path = SERVER_DIR / f"{user_id}_encrypted_input"

    # Read the files (Evaluation key + Encrypted symptoms) using the above paths
    with encrypted_input_path.open("rb") as encrypted_output_file, evaluation_key_path.open(
        "rb"
    ) as evaluation_key_file:
        encrypted_output = encrypted_output_file.read()
        evaluation_key = evaluation_key_file.read()

    # Run the FHE execution
    start = time.time()
    encrypted_output = FHE_SERVER.run(encrypted_output, evaluation_key)
    assert isinstance(encrypted_output, bytes)
    fhe_execution_time = round(time.time() - start, 2)

    # Retrieve the encrypted output path
    encrypted_output_path = SERVER_DIR / f"{user_id}_encrypted_output"

    # Write the file using the above path
    with encrypted_output_path.open("wb") as f:
        f.write(encrypted_output)

    return JSONResponse(content=fhe_execution_time)


@app.post("/get_output")
def get_output(user_id: str = Form()):
    """Retrieve the encrypted output from the server."""

    print("\nGet the output from the server ............\n")

    # Path where the encrypted output is saved
    encrypted_output_path = SERVER_DIR / f"{user_id}_encrypted_output"

    # Read the file using the above path
    with encrypted_output_path.open("rb") as f:
        encrypted_output = f.read()

    time.sleep(1)

    # Send the encrypted output
    return Response(encrypted_output)
