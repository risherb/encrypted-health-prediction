import subprocess
import time
from typing import Dict, List, Tuple

import gradio as gr  # pylint: disable=import-error
import numpy as np
import pandas as pd
import requests
from symptoms_categories import SYMPTOMS_LIST
from utils import (
    CLIENT_DIR,
    CURRENT_DIR,
    DEPLOYMENT_DIR,
    INPUT_BROWSER_LIMIT,
    KEYS_DIR,
    SERVER_URL,
    TARGET_COLUMNS,
    TRAINING_FILENAME,
    clean_directory,
    get_disease_name,
    load_data,
    pretty_print,
)

from concrete.ml.deployment import FHEModelClient

subprocess.Popen(["uvicorn", "server:app"], cwd=CURRENT_DIR)
time.sleep(3)

# pylint: disable=c-extension-no-member,invalid-name


def is_none(obj) -> bool:
    """
    Check if the object is None.

    Args:
        obj (any): The input to be checked.

    Returns:
        bool: True if the object is None or empty, False otherwise.
    """
    return obj is None or (obj is not None and len(obj) < 1)


def display_default_symptoms_fn(default_disease: str) -> Dict:
    """
    Displays the symptoms of a given existing disease.

    Args:
        default_disease (str): Disease
    Returns:
        Dict: The according symptoms
    """
    df = pd.read_csv(TRAINING_FILENAME)
    df_filtred = df[df[TARGET_COLUMNS[1]] == default_disease]

    return {
        default_symptoms: gr.update(
            visible=True,
            value=pretty_print(
                df_filtred.columns[df_filtred.eq(1).any()].to_list(), delimiter=", "
            ),
        )
    }


def get_user_symptoms_from_checkboxgroup(checkbox_symptoms: List) -> np.array:
    """
    Convert the user symptoms into a binary vector representation.

    Args:
        checkbox_symptoms (List): A list of user symptoms.

    Returns:
        np.array: A binary vector representing the user's symptoms.

    Raises:
        KeyError: If a provided symptom is not recognized as a valid symptom.

    """
    symptoms_vector = {key: 0 for key in valid_symptoms}
    for pretty_symptom in checkbox_symptoms:
        original_symptom = "_".join((pretty_symptom.lower().split(" ")))
        if original_symptom not in symptoms_vector.keys():
            raise KeyError(
                f"The symptom '{original_symptom}' you provided is not recognized as a valid "
                f"symptom.\nHere is the list of valid symptoms: {symptoms_vector}"
            )
        symptoms_vector[original_symptom] = 1

    user_symptoms_vect = np.fromiter(symptoms_vector.values(), dtype=float)[np.newaxis, :]

    assert all(value == 0 or value == 1 for value in user_symptoms_vect.flatten())

    return user_symptoms_vect


def get_features_fn(*checked_symptoms: Tuple[str]) -> Dict:
    """
    Get vector features based on the selected symptoms.

    Args:
        checked_symptoms (Tuple[str]): User symptoms

    Returns:
        Dict: The encoded user vector symptoms.
    """
    if not any(lst for lst in checked_symptoms if lst):
        return {
            error_box1: gr.update(visible=True, value="⚠️ Please provide your chief complaints."),
        }

    if len(pretty_print(checked_symptoms)) < 5:
        print("Provide at least 5 symptoms.")
        return {
            error_box1: gr.update(visible=True, value="⚠️ Provide at least 5 symptoms"),
            one_hot_vect: None,
        }

    return {
        error_box1: gr.update(visible=False),
        one_hot_vect: gr.update(
            visible=False,
            value=get_user_symptoms_from_checkboxgroup(pretty_print(checked_symptoms)),
        ),
        submit_btn: gr.update(value="Data submitted ✅"),
    }


def key_gen_fn(user_symptoms: List[str]) -> Dict:
    """
    Generate keys for a given user.

    Args:
        user_symptoms (List[str]): The vector symptoms provided by the user.

    Returns:
        dict: A dictionary containing the generated keys and related information.

    """
    clean_directory()

    if is_none(user_symptoms):
        print("Error: Please submit your symptoms or select a default disease.")
        return {
            error_box2: gr.update(visible=True, value="⚠️ Please submit your symptoms first."),
        }

    # Generate a random user ID
    user_id = np.random.randint(0, 2**32)
    print(f"Your user ID is: {user_id}....")

    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    # Creates the private and evaluation keys on the client side
    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)

    # Save the evaluation key
    evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
    with evaluation_key_path.open("wb") as f:
        f.write(serialized_evaluation_keys)

    serialized_evaluation_keys_shorten_hex = serialized_evaluation_keys.hex()[:INPUT_BROWSER_LIMIT]

    return {
        error_box2: gr.update(visible=False),
        key_box: gr.update(visible=False, value=serialized_evaluation_keys_shorten_hex),
        user_id_box: gr.update(visible=False, value=user_id),
        key_len_box: gr.update(
            visible=False, value=f"{len(serialized_evaluation_keys) / (10**6):.2f} MB"
        ),
        gen_key_btn: gr.update(value="Keys have been generated ✅")
    }


def encrypt_fn(user_symptoms: np.ndarray, user_id: str) -> None:
    """
    Encrypt the user symptoms vector in the `Client Side`.

    Args:
        user_symptoms (List[str]): The vector symptoms provided by the user
        user_id (user): The current user's ID
    """

    if is_none(user_id) or is_none(user_symptoms):
        print("Error in encryption step: Provide your symptoms and generate the evaluation keys.")
        return {
            error_box3: gr.update(
                visible=True,
                value="⚠️ Please ensure that your symptoms have been submitted and "
                "that you have generated the evaluation key.",
            )
        }

    # Retrieve the client API
    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    user_symptoms = np.fromstring(user_symptoms[2:-2], dtype=int, sep=".").reshape(1, -1)
    # quant_user_symptoms = client.model.quantize_input(user_symptoms)

    encrypted_quantized_user_symptoms = client.quantize_encrypt_serialize(user_symptoms)
    assert isinstance(encrypted_quantized_user_symptoms, bytes)
    encrypted_input_path = KEYS_DIR / f"{user_id}/encrypted_input"

    with encrypted_input_path.open("wb") as f:
        f.write(encrypted_quantized_user_symptoms)

    encrypted_quantized_user_symptoms_shorten_hex = encrypted_quantized_user_symptoms.hex()[
        :INPUT_BROWSER_LIMIT
    ]

    return {
        error_box3: gr.update(visible=False),
        one_hot_vect_box: gr.update(visible=True, value=user_symptoms),
        enc_vect_box: gr.update(visible=True, value=encrypted_quantized_user_symptoms_shorten_hex),
    }


def send_input_fn(user_id: str, user_symptoms: np.ndarray) -> Dict:
    """Send the encrypted data and the evaluation key to the server.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box4: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the symptoms have been submitted and the evaluation "
                "key has been generated before sending the data to the server.",
            )
        }

    evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
    encrypted_input_path = KEYS_DIR / f"{user_id}/encrypted_input"

    if not evaluation_key_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: "
            f"The key has been generated correctly - {evaluation_key_path.is_file()=}"
        )

        return {
            error_box4: gr.update(visible=True, value="⚠️ Please generate the private key first.")
        }

    if not encrypted_input_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: The data has not been encrypted "
            f"correctly on the client side - {encrypted_input_path.is_file()=}"
        )
        return {
            error_box4: gr.update(
                visible=True,
                value="⚠️ Please encrypt the data with the private key first.",
            ),
        }

    # Define the data and files to post
    data = {
        "user_id": user_id,
        "input": user_symptoms,
    }

    files = [
        ("files", open(encrypted_input_path, "rb")),
        ("files", open(evaluation_key_path, "rb")),
    ]

    # Send the encrypted input and evaluation key to the server
    url = SERVER_URL + "send_input"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        print(f"Sending Data: {response.ok=}")
    return {
        error_box4: gr.update(visible=False),
        srv_resp_send_data_box: "Data sent",
    }


def run_fhe_fn(user_id: str) -> Dict:
    """Send the encrypted input and the evaluation key to the server.

    Args:
        user_id (int): The current user's ID.
    """
    if is_none(user_id):
        return {
            error_box5: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the symptoms have been submitted, the evaluation "
                "key has been generated and the server received the data "
                "before processing the data.",
            ),
            fhe_execution_time_box: None,
        }

    data = {
        "user_id": user_id,
    }

    url = SERVER_URL + "run_fhe"

    with requests.post(
        url=url,
        data=data,
    ) as response:
        if not response.ok:
            return {
                error_box5: gr.update(
                    visible=True,
                    value=(
                        "⚠️ An error occurred on the Server Side. "
                        "Please check connectivity and data transmission."
                    ),
                ),
                fhe_execution_time_box: gr.update(visible=False),
            }
        else:
            time.sleep(1)
            print(f"response.ok: {response.ok}, {response.json()} - Computed")

    return {
        error_box5: gr.update(visible=False),
        fhe_execution_time_box: gr.update(visible=True, value=f"{response.json():.2f} seconds"),
    }


def get_output_fn(user_id: str, user_symptoms: np.ndarray) -> Dict:
    """Retreive the encrypted data from the server.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box6: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the server has successfully processed and transmitted the data to the client.",
            )
        }

    data = {
        "user_id": user_id,
    }

    # Retrieve the encrypted output
    url = SERVER_URL + "get_output"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            print(f"Receive Data: {response.ok=}")

            encrypted_output = response.content

            # Save the encrypted output to bytes in a file as it is too large to pass through
            # regular Gradio buttons (see https://github.com/gradio-app/gradio/issues/1877)
            encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"

            with encrypted_output_path.open("wb") as f:
                f.write(encrypted_output)
    return {error_box6: gr.update(visible=False), srv_resp_retrieve_data_box: "Data received"}


def decrypt_fn(
    user_id: str, user_symptoms: np.ndarray, *checked_symptoms, threshold: int = 0.5
) -> Dict:
    """Dencrypt the data on the `Client Side`.

    Args:
        user_id (str): The current user's ID
        user_symptoms (np.ndarray): The user symptoms
        threshold (float): Probability confidence threshold

    Returns:
        Decrypted output
    """

    if is_none(user_id) or is_none(user_symptoms):
        return {
            error_box7: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the client has successfully received the data from the server.",
            )
        }

    # Get the encrypted output path
    encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"

    if not encrypted_output_path.is_file():
        print("Error in decryption step: Please run the FHE execution, first.")
        return {
            error_box7: gr.update(
                visible=True,
                value="⚠️ Please ensure that: \n"
                "- the connectivity \n"
                "- the symptoms have been submitted \n"
                "- the evaluation key has been generated \n"
                "- the server processed the encrypted data \n"
                "- the Client received the data from the Server before decrypting the prediction",
            ),
            decrypt_box: None,
        }

    # Load the encrypted output as bytes
    with encrypted_output_path.open("rb") as f:
        encrypted_output = f.read()

    # Retrieve the client API
    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    # Deserialize, decrypt and post-process the encrypted output
    output = client.deserialize_decrypt_dequantize(encrypted_output)

    top3_diseases = np.argsort(output.flatten())[-3:][::-1]
    top3_proba = output[0][top3_diseases]

    out = ""

    if top3_proba[0] < threshold or abs(top3_proba[0] - top3_proba[1]) < 0.1:
        out = (
            "⚠️ The prediction appears uncertain; including more symptoms "
            "may improve the results.\n\n"
        )

    out = (
        f"{out}Given the symptoms you provided: "
        f"{pretty_print(checked_symptoms, case_conversion=str.capitalize, delimiter=', ')}\n\n"
        "Here are the top3 predictions:\n\n"
        f"1. « {get_disease_name(top3_diseases[0])} » with a probability of {top3_proba[0]:.2%}\n"
        f"2. « {get_disease_name(top3_diseases[1])} » with a probability of {top3_proba[1]:.2%}\n"
        f"3. « {get_disease_name(top3_diseases[2])} » with a probability of {top3_proba[2]:.2%}\n"
    )

    return {
        error_box7: gr.update(visible=False),
        decrypt_box: out,
        submit_btn: gr.update(value="Submit"),
    }


def reset_fn():
    """Reset the space and clear all the box outputs."""

    clean_directory()

    return {
        one_hot_vect: None,
        one_hot_vect_box: None,
        enc_vect_box: gr.update(visible=True, value=None),
        quant_vect_box: gr.update(visible=False, value=None),
        user_id_box: gr.update(visible=False, value=None),
        default_symptoms: gr.update(visible=True, value=None),
        default_disease_box: gr.update(visible=True, value=None),
        key_box: gr.update(visible=True, value=None),
        key_len_box: gr.update(visible=False, value=None),
        fhe_execution_time_box: gr.update(visible=True, value=None),
        decrypt_box: None,
        submit_btn: gr.update(value="Submit"),
        error_box7: gr.update(visible=False),
        error_box1: gr.update(visible=False),
        error_box2: gr.update(visible=False),
        error_box3: gr.update(visible=False),
        error_box4: gr.update(visible=False),
        error_box5: gr.update(visible=False),
        error_box6: gr.update(visible=False),
        srv_resp_send_data_box: None,
        srv_resp_retrieve_data_box: None,
        **{box: None for box in check_boxes},
    }


if __name__ == "__main__":

    print("Starting demo ...")

    clean_directory()

    (X_train, X_test), (y_train, y_test), valid_symptoms, diseases = load_data()

    with gr.Blocks() as demo:

        # Link + images
        gr.Markdown()
        gr.Markdown()
        gr.Markdown("""<h2 align="center">Health Prediction On Encrypted Data Using FHE</h2>""")
        gr.Markdown()
        gr.Markdown(
            """
            <p align="center">
                <a href="https://github.com/risherb/FHE-Based-Health-Prediction-"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197972109-faaaff3e-10e2-4ab6-80f5-7531f7cfb08f.png">Github</a>
                —
                <a href="https://docs.zama.ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197976802-fddd34c5-f59a-48d0-9bff-7ad1b00cb1fb.png">Documentation</a>
                —
                <a href="https://linkedin.com/in/rishabhnshetty" style="margin: 0 15px;"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="https://user-images.githubusercontent.com/5758427/197975044-bab9d199-e120-433b-b3be-abd73b211a54.png">@rishabhnshetty</a>

            """)
        gr.Markdown()
        gr.Markdown("## Notes")
        gr.Markdown(
            """
            - The private key is used to encrypt and decrypt the data and shall never be shared.
            - The evaluation key is a public key that the server needs to process encrypted data.
            """
        )

        # ------------------------- Step 1 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 1: Select chief complaints")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Client Side</span>")
        gr.Markdown("Select at least 5 chief complaints from the list below.")

        # Step 1.1: Provide symptoms
        check_boxes = []
        with gr.Row():
            with gr.Column():
                for category in SYMPTOMS_LIST[:3]:
                    with gr.Accordion(pretty_print(category.keys()), open=False):
                        check_box = gr.CheckboxGroup(pretty_print(category.values()), show_label=0)
                        check_boxes.append(check_box)
            with gr.Column():
                for category in SYMPTOMS_LIST[3:6]:
                    with gr.Accordion(pretty_print(category.keys()), open=False):
                        check_box = gr.CheckboxGroup(pretty_print(category.values()), show_label=0)
                        check_boxes.append(check_box)
            with gr.Column():
                for category in SYMPTOMS_LIST[6:]:
                    with gr.Accordion(pretty_print(category.keys()), open=False):
                        check_box = gr.CheckboxGroup(pretty_print(category.values()), show_label=0)
                        check_boxes.append(check_box)

        error_box1 = gr.Textbox(label="Error ❌", visible=False)

        # Default disease, picked from the dataframe
        gr.Markdown(
            "You can choose an **existing disease** and explore its associated symptoms.",
            visible=False,
        )

        with gr.Row():
            with gr.Column(scale=2):
                default_disease_box = gr.Dropdown(sorted(diseases), label="Diseases", visible=False)
            with gr.Column(scale=5):
                default_symptoms = gr.Textbox(label="Related Symptoms:", visible=False)
        # User vector symptoms encoded in oneHot representation
        one_hot_vect = gr.Textbox(visible=False)
        # Submit botton
        submit_btn = gr.Button("Submit")
        # Clear botton
        clear_button = gr.Button("Reset Space 🔁", visible=False)

        default_disease_box.change(
            fn=display_default_symptoms_fn, inputs=[default_disease_box], outputs=[default_symptoms]
        )

        submit_btn.click(
            fn=get_features_fn,
            inputs=[*check_boxes],
            outputs=[one_hot_vect, error_box1, submit_btn],
        )

        # ------------------------- Step 2 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 2: Encrypt data")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Client Side</span>")
        # Step 2.1: Key generation
        gr.Markdown(
            "### Key Generation\n\n"
            "In FHE schemes, a secret (enc/dec)ryption keys are generated for encrypting and decrypting data owned by the client. \n\n"
            "Additionally, a public evaluation key is generated, enabling external entities to perform homomorphic operations on encrypted data, without the need to decrypt them. \n\n"
            "The evaluation key will be transmitted to the server for further processing."
        )

        gen_key_btn = gr.Button("Generate the private and evaluation keys.")
        error_box2 = gr.Textbox(label="Error ❌", visible=False)
        user_id_box = gr.Textbox(label="User ID:", visible=False)
        key_len_box = gr.Textbox(label="Evaluation Key Size:", visible=False)
        key_box = gr.Textbox(label="Evaluation key (truncated):", max_lines=3, visible=False)

        gen_key_btn.click(
            key_gen_fn,
            inputs=one_hot_vect,
            outputs=[
                key_box,
                user_id_box,
                key_len_box,
                error_box2,
                gen_key_btn,
            ],
        )

        # Step 2.2: Encrypt data locally
        gr.Markdown("### Encrypt the data")
        encrypt_btn = gr.Button("Encrypt the data using the private secret key")
        error_box3 = gr.Textbox(label="Error ❌", visible=False)
        quant_vect_box = gr.Textbox(label="Quantized Vector:", visible=False)

        with gr.Row():
            with gr.Column():
                one_hot_vect_box = gr.Textbox(label="User Symptoms Vector:", max_lines=10)
            with gr.Column():
                enc_vect_box = gr.Textbox(label="Encrypted Vector:", max_lines=10)

        encrypt_btn.click(
            encrypt_fn,
            inputs=[one_hot_vect, user_id_box],
            outputs=[
                one_hot_vect_box,
                enc_vect_box,
                error_box3,
            ],
        )
        # Step 2.3: Send encrypted data to the server
        gr.Markdown(
            "### Send the encrypted data to the <span style='color:grey'>Server Side</span>"
        )
        error_box4 = gr.Textbox(label="Error ❌", visible=False)

        # with gr.Row().style(equal_height=False):
        with gr.Row():
            with gr.Column(scale=4):
                send_input_btn = gr.Button("Send data")
            with gr.Column(scale=1):
                srv_resp_send_data_box = gr.Checkbox(label="Data Sent", show_label=False)

        send_input_btn.click(
            send_input_fn,
            inputs=[user_id_box, one_hot_vect],
            outputs=[error_box4, srv_resp_send_data_box],
        )

        # ------------------------- Step 3 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 3: Run the FHE evaluation")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Server Side</span>")

        run_fhe_btn = gr.Button("Run the FHE evaluation")
        error_box5 = gr.Textbox(label="Error ❌", visible=False)
        fhe_execution_time_box = gr.Textbox(label="Total FHE Execution Time:", visible=True)
        run_fhe_btn.click(
            run_fhe_fn,
            inputs=[user_id_box],
            outputs=[fhe_execution_time_box, error_box5],
        )

        # ------------------------- Step 4 -------------------------
        gr.Markdown("\n")
        gr.Markdown("## Step 4: Decrypt the data")
        gr.Markdown("<hr />")
        gr.Markdown("<span style='color:grey'>Client Side</span>")
        gr.Markdown(
            "### Get the encrypted data from the <span style='color:grey'>Server Side</span>"
        )

        error_box6 = gr.Textbox(label="Error ❌", visible=False)

        # Step 4.1: Data transmission
        # with gr.Row().style(equal_height=True):
        with gr.Row():
            with gr.Column(scale=4):
                get_output_btn = gr.Button("Get data")
            with gr.Column(scale=1):
                srv_resp_retrieve_data_box = gr.Checkbox(label="Data Received", show_label=False)

        get_output_btn.click(
            get_output_fn,
            inputs=[user_id_box, one_hot_vect],
            outputs=[srv_resp_retrieve_data_box, error_box6],
        )

        # Step 4.1: Data transmission
        gr.Markdown("### Decrypt the output")
        decrypt_btn = gr.Button("Decrypt the output using the private secret key")
        error_box7 = gr.Textbox(label="Error ❌", visible=False)
        decrypt_box = gr.Textbox(label="Decrypted Output:")

        decrypt_btn.click(
            decrypt_fn,
            inputs=[user_id_box, one_hot_vect, *check_boxes],
            outputs=[decrypt_box, error_box7, submit_btn],
        )

        # ------------------------- End -------------------------

        gr.Markdown(
            """The app was built by Rishabh Rahul & Raeez for their Minor Project.
            """
        )

        gr.Markdown("\n\n")

        clear_button.click(
            reset_fn,
            outputs=[
                one_hot_vect_box,
                one_hot_vect,
                submit_btn,
                error_box1,
                error_box2,
                error_box3,
                error_box4,
                error_box5,
                error_box6,
                error_box7,
                default_disease_box,
                default_symptoms,
                user_id_box,
                key_len_box,
                key_box,
                quant_vect_box,
                enc_vect_box,
                srv_resp_send_data_box,
                srv_resp_retrieve_data_box,
                fhe_execution_time_box,
                decrypt_box,
                *check_boxes,
            ],
        )

        demo.launch()
