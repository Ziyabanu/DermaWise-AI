from fastapi import FastAPI, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware


import tensorflow as tf

import numpy as np

import cv2


from PIL import Image

from io import BytesIO

from pathlib import Path


import base64
import json

# =========================================================

# FastAPI

# =========================================================


app = FastAPI(title="DermaWise AI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================

# Paths

# =========================================================


BASE_DIR = Path(__file__).resolve().parent


LESION_MODEL_PATH = BASE_DIR.parent / "Model" / "dermawise_grouped_finetuned_best.keras"


COMMON_MODEL_PATH = (
    BASE_DIR.parent / "CommonSkinModel" / "dermawise_common_skin_final.keras"
)
BROAD_MODEL_PATH = (
    BASE_DIR.parent / "Model" / "Broad114" / "dermawise_kd_student_best.keras"
)

BROAD_MAPPING_PATH = (
    BASE_DIR.parent / "Model" / "Broad114" / "fitzpatrick114_class_mapping.json"
)

# =========================================================

# Load both models

# =========================================================


print("Loading DermaWise Lesion model...")


lesion_model = tf.keras.models.load_model(LESION_MODEL_PATH)


print("Lesion model loaded successfully!")


print("Loading DermaWise Common Skin model...")


common_model = tf.keras.models.load_model(COMMON_MODEL_PATH)


print("Common Skin model loaded successfully!")

print("Loading DermaWise Experimental 114-Condition model...")
broad_model = tf.keras.models.load_model(
    BROAD_MODEL_PATH,
    compile=False,
)

with open(BROAD_MAPPING_PATH, "r", encoding="utf-8") as mapping_file:
    broad_mapping_raw = json.load(mapping_file)

# Support either {"0": "label", ...} or a JSON list.
if isinstance(broad_mapping_raw, dict):
    BROAD_CLASS_NAMES = [broad_mapping_raw[str(i)] for i in range(114)]
else:
    BROAD_CLASS_NAMES = list(broad_mapping_raw)

if len(BROAD_CLASS_NAMES) != 114:
    raise ValueError("Broad114 class mapping must contain exactly 114 labels.")

print("Experimental 114-Condition model loaded successfully!")


# =========================================================

# LESION MODEL INFORMATION

# =========================================================


LESION_CLASS_CODES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


LESION_CLASS_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}


LESION_GUIDANCE = {
    "akiec": {
        "description": (
            "Actinic keratosis is a sun-related skin lesion that can "
            "sometimes progress to a more serious condition."
        ),
        "guidance": (
            "Consider evaluation by a qualified healthcare professional, "
            "especially if the area is changing, persistent, bleeding, "
            "or uncomfortable."
        ),
    },
    "bcc": {
        "description": (
            "Basal cell carcinoma is a type of skin cancer that commonly "
            "develops on areas of skin exposed to the sun."
        ),
        "guidance": (
            "Professional dermatologic evaluation is recommended. "
            "An AI result alone cannot confirm or rule out basal cell carcinoma."
        ),
    },
    "bkl": {
        "description": (
            "Benign keratosis represents a group of generally non-cancerous "
            "keratotic skin lesions."
        ),
        "guidance": (
            "Consider professional evaluation if the lesion is new, changing, "
            "irritated, bleeding, or otherwise concerning."
        ),
    },
    "df": {
        "description": (
            "Dermatofibroma is usually a benign skin growth that often "
            "appears as a small, firm bump."
        ),
        "guidance": (
            "If the area changes noticeably, becomes painful, bleeds, "
            "or causes concern, consider professional evaluation."
        ),
    },
    "mel": {
        "description": (
            "Melanoma is a serious form of skin cancer involving "
            "pigment-producing cells."
        ),
        "guidance": (
            "Prompt professional evaluation is important for a suspicious "
            "or changing pigmented lesion. This model result must not be "
            "used to diagnose melanoma."
        ),
    },
    "nv": {
        "description": (
            "Melanocytic nevi are commonly known as moles and are usually benign."
        ),
        "guidance": (
            "Consider professional evaluation if a mole changes in size, "
            "shape, color, appearance, or develops symptoms such as bleeding "
            "or persistent itching."
        ),
    },
    "vasc": {
        "description": (
            "Vascular lesions involve blood vessels and include several "
            "different types of skin findings."
        ),
        "guidance": (
            "Consider professional evaluation if the area is new, changing, "
            "bleeding, painful, or otherwise concerning."
        ),
    },
}


# =========================================================

# COMMON SKIN MODEL INFORMATION

# IMPORTANT:

# Keep this order exactly the same as model training.

# =========================================================


COMMON_CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Folliculitis",
    "Psoriasis",
]


COMMON_GUIDANCE = {
    "Acne": {
        "description": (
            "The image shows visual features that the model associates "
            "with acne-like skin findings."
        ),
        "guidance": (
            "Avoid picking or squeezing affected areas. Gentle cleansing "
            "and non-comedogenic skin-care products may be helpful. "
            "Persistent, painful, or scarring acne should be evaluated "
            "by a qualified healthcare professional."
        ),
    },
    "Eczema": {
        "description": (
            "The image shows visual features that the model associates "
            "with eczema-like skin findings."
        ),
        "guidance": (
            "Gentle fragrance-free moisturizers and avoiding known skin "
            "irritants may be helpful. Seek professional evaluation if "
            "the area is severe, widespread, persistent, painful, or "
            "shows possible signs of infection."
        ),
    },
    "Folliculitis": {
        "description": (
            "The image shows visual features that the model associates "
            "with folliculitis-like skin findings."
        ),
        "guidance": (
            "Avoid unnecessary friction, scratching, or shaving over an "
            "irritated area. Professional evaluation is appropriate if "
            "the problem is painful, spreading, recurrent, or accompanied "
            "by fever or other concerning symptoms."
        ),
    },
    "Psoriasis": {
        "description": (
            "The image shows visual features that the model associates "
            "with psoriasis-like skin findings."
        ),
        "guidance": (
            "Gentle moisturizing and avoiding known skin triggers may be "
            "helpful. Persistent, widespread, painful, or worsening areas "
            "should be evaluated by a qualified healthcare professional."
        ),
    },
}


# =========================================================

# Thresholds

# =========================================================


LESION_CONFIDENCE_THRESHOLD = 0.60


COMMON_CONFIDENCE_THRESHOLD = 0.60


# =========================================================

# LESION GRAD-CAM SETUP

# =========================================================


lesion_base_model = lesion_model.get_layer("efficientnetb0")


lesion_last_conv_layer = lesion_base_model.get_layer("top_activation")


lesion_grad_model = tf.keras.Model(
    inputs=lesion_base_model.input,
    outputs=[
        lesion_last_conv_layer.output,
        lesion_base_model.output,
    ],
)


# =========================================================

# COMMON SKIN GRAD-CAM SETUP

# =========================================================


common_base_model = common_model.get_layer("efficientnetb0")

common_last_conv_layer = common_base_model.get_layer("block6a_expand_activation")


common_grad_model = tf.keras.Model(
    inputs=common_base_model.input,
    outputs=[
        common_last_conv_layer.output,
        common_base_model.output,
    ],
)


# =========================================================

# Generic Grad-CAM generator

# =========================================================


def generate_gradcam(
    model,
    grad_model,
    img_array,
    predicted_index,
):
    """
    Generate standard Grad-CAM for the selected predicted class.

    This changes only the explainability heatmap, not the prediction.
    """
    x = model.layers[1](img_array, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(x, training=False)
        tape.watch(conv_outputs)

        x_head = base_outputs
        for layer in model.layers[3:]:
            try:
                x_head = layer(x_head, training=False)
            except TypeError:
                x_head = layer(x_head)

        class_output = x_head[:, predicted_index]

    gradients = tape.gradient(class_output, conv_outputs)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1,
    )

    heatmap = tf.nn.relu(heatmap)
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + tf.keras.backend.epsilon()),
        heatmap,
    )

    return heatmap.numpy()


# =========================================================

# Grad-CAM overlay

# =========================================================


def create_gradcam_overlay(
    original_image,
    heatmap,
):
    """
    Create a visible Grad-CAM overlay on the original image.

    The visualization enhances genuine Grad-CAM activations
    without changing their spatial location or the model prediction.
    """

    original = original_image.convert("RGB")
    width, height = original.size

    heatmap_image = Image.fromarray(np.uint8(heatmap * 255))

    heatmap_image = heatmap_image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    heatmap_array = (
        np.array(
            heatmap_image,
            dtype=np.float32,
        )
        / 255.0
    )

    original_array = np.array(
        original,
        dtype=np.float32,
    )

    red = np.clip(
        heatmap_array * 2,
        0,
        1,
    )

    green = np.clip(
        heatmap_array * 1.2,
        0,
        1,
    )

    blue = np.zeros_like(heatmap_array)

    color_heatmap = (
        np.stack(
            [
                red,
                green,
                blue,
            ],
            axis=-1,
        )
        * 255
    )

    # Improve visibility of genuine Grad-CAM activations only.
    # This does not move the heatmap or change model predictions.
    visible_heatmap = np.power(heatmap_array, 0.5)

    alpha = visible_heatmap[..., np.newaxis] * 0.65

    overlay = original_array * (1 - alpha) + color_heatmap * alpha

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(overlay)


# =========================================================

# Base64 conversion

# =========================================================


def image_to_base64(image):

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


# =========================================================

# Image Quality Check

# =========================================================


def check_image_quality(image):

    gray = np.array(
        image.convert("L"),
        dtype=np.uint8,
    )

    brightness = float(np.mean(gray))

    if brightness < 40:

        return {
            "passed": False,
            "reason": (
                "Image is too dark. Please capture the image " "in better lighting."
            ),
            "brightness": round(
                brightness,
                2,
            ),
        }

    if brightness > 220:

        return {
            "passed": False,
            "reason": ("Image is too bright. Please reduce the lighting or glare."),
            "brightness": round(
                brightness,
                2,
            ),
        }

    sharpness_score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    if sharpness_score < 4:

        return {
            "passed": False,
            "reason": ("Image appears too blurry. " "Please capture a clearer image."),
            "brightness": round(
                brightness,
                2,
            ),
            "sharpness": round(
                float(sharpness_score),
                2,
            ),
        }

    return {
        "passed": True,
        "reason": "Image quality is acceptable.",
        "brightness": round(
            brightness,
            2,
        ),
        "sharpness": round(
            float(sharpness_score),
            2,
        ),
    }


# =========================================================

# Read and validate uploaded image

# =========================================================


async def read_uploaded_image(file):

    contents = await file.read()

    try:

        image = Image.open(BytesIO(contents)).convert("RGB")

        return image, None

    except Exception:

        return None, {
            "quality_passed": False,
            "quality_message": (
                "Invalid or corrupted image. "
                "Please upload a valid JPEG, PNG, or WebP image."
            ),
            "quality_details": {
                "passed": False,
                "reason": ("The uploaded file could not be read as an image."),
            },
        }


# =========================================================

# Prepare image

# =========================================================


def prepare_image(original_image):

    model_image = original_image.resize((224, 224))

    img_array = np.array(
        model_image,
        dtype=np.float32,
    )

    img_array = np.expand_dims(
        img_array,
        axis=0,
    )

    return img_array


def prepare_broad_image(original_image):
    model_image = original_image.resize((260, 260))

    img_array = np.array(model_image, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# =========================================================

# HOME

# =========================================================


@app.get("/")
def home():

    return {
        "message": "DermaWise AI API is running",
        "available_analysis": [
            "Lesion Analysis",
            "Common Skin Analysis",
        ],
    }


# =========================================================

# EXISTING LESION PREDICTION

# =========================================================


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    original_image, error = await read_uploaded_image(file)

    if error:

        return error

    quality = check_image_quality(original_image)

    if not quality["passed"]:

        return {
            "quality_passed": False,
            "quality_message": quality["reason"],
            "quality_details": quality,
        }

    img_array = prepare_image(original_image)

    predictions = lesion_model.predict(
        img_array,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_code = LESION_CLASS_CODES[predicted_index]

    confidence = float(predictions[predicted_index])

    probabilities = {}

    for i, code in enumerate(LESION_CLASS_CODES):

        probabilities[LESION_CLASS_NAMES[code]] = round(
            float(predictions[i]) * 100,
            2,
        )

    uncertain = confidence < LESION_CONFIDENCE_THRESHOLD

    result = "Uncertain" if uncertain else LESION_CLASS_NAMES[predicted_code]

    heatmap = generate_gradcam(
        lesion_model,
        lesion_grad_model,
        img_array,
        predicted_index,
    )

    gradcam_image = create_gradcam_overlay(
        original_image,
        heatmap,
    )

    gradcam_base64 = image_to_base64(gradcam_image)

    return {
        "analysis_type": "lesion",
        "prediction": result,
        "predicted_code": predicted_code,
        "predicted_class": LESION_CLASS_NAMES[predicted_code],
        "description": LESION_GUIDANCE[predicted_code]["description"],
        "guidance": LESION_GUIDANCE[predicted_code]["guidance"],
        "confidence": round(
            confidence * 100,
            2,
        ),
        "uncertain": uncertain,
        "probabilities": probabilities,
        "gradcam_image": gradcam_base64,
        "quality_passed": True,
        "quality_message": quality["reason"],
        "quality_details": quality,
    }


# =========================================================

# NEW COMMON SKIN PREDICTION

# =========================================================


@app.post("/predict-common")
async def predict_common_skin(file: UploadFile = File(...)):

    original_image, error = await read_uploaded_image(file)

    if error:

        return error

    # -----------------------------------------------------

    # Image quality

    # -----------------------------------------------------

    quality = check_image_quality(original_image)

    if not quality["passed"]:

        return {
            "analysis_type": "common_skin",
            "quality_passed": False,
            "quality_message": quality["reason"],
            "quality_details": quality,
        }

    # -----------------------------------------------------

    # Preprocess

    # -----------------------------------------------------

    img_array = prepare_image(original_image)

    # -----------------------------------------------------

    # Prediction

    # -----------------------------------------------------

    predictions = common_model.predict(
        img_array,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = COMMON_CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index])

    # -----------------------------------------------------

    # All four class scores

    # -----------------------------------------------------

    probabilities = {}

    for i, class_name in enumerate(COMMON_CLASS_NAMES):

        probabilities[class_name] = round(
            float(predictions[i]) * 100,
            2,
        )

    # -----------------------------------------------------

    # Uncertainty

    # -----------------------------------------------------

    uncertain = confidence < COMMON_CONFIDENCE_THRESHOLD

    result = "Uncertain" if uncertain else predicted_class

    # -----------------------------------------------------

    # Grad-CAM

    # -----------------------------------------------------

    heatmap = generate_gradcam(
        common_model,
        common_grad_model,
        img_array,
        predicted_index,
    )

    gradcam_image = create_gradcam_overlay(
        original_image,
        heatmap,
    )

    gradcam_base64 = image_to_base64(gradcam_image)

    # -----------------------------------------------------

    # Guidance

    #

    # If uncertain, do NOT present class-specific guidance

    # as though it were a reliable classification.

    # -----------------------------------------------------

    if uncertain:

        description = (
            "The model did not reach the confidence threshold "
            "required to provide a reliable category."
        )

        guidance = (
            "The highest-scoring category is shown for transparency, "
            "but this result should be treated as uncertain. Consider "
            "a qualified healthcare professional if the skin concern "
            "is persistent, worsening, painful, widespread, or otherwise concerning."
        )

    else:

        description = COMMON_GUIDANCE[predicted_class]["description"]

        guidance = COMMON_GUIDANCE[predicted_class]["guidance"]

    # -----------------------------------------------------

    # Response

    # -----------------------------------------------------

    return {
        "analysis_type": "common_skin",
        "prediction": result,
        "predicted_class": predicted_class,
        "description": description,
        "guidance": guidance,
        "confidence": round(
            confidence * 100,
            2,
        ),
        "uncertain": uncertain,
        "probabilities": probabilities,
        "gradcam_image": gradcam_base64,
        "quality_passed": True,
        "quality_message": quality["reason"],
        "quality_details": quality,
        "supported_categories": [
            "Acne",
            "Eczema",
            "Folliculitis",
            "Psoriasis",
        ],
        "disclaimer": (
            "DermaWise is a research prototype and not a medical "
            "diagnostic system. Model scores are not clinical probabilities."
        ),
    }

    # =========================================================


# UNIFIED DERMAWISE ANALYSIS

# =========================================================


@app.post("/analyze")
async def analyze_skin(file: UploadFile = File(...)):

    # Read uploaded image

    original_image, error = await read_uploaded_image(file)

    if error:

        return error

    # Check image quality

    quality = check_image_quality(original_image)

    if not quality["passed"]:

        return {
            "analysis_type": "unified",
            "quality_passed": False,
            "quality_message": quality["reason"],
            "quality_details": quality,
        }

    # Prepare image

    img_array = prepare_image(original_image)

    # -----------------------------------------------------

    # LESION MODEL

    # -----------------------------------------------------

    lesion_predictions = lesion_model.predict(img_array, verbose=0)[0]

    lesion_index = int(np.argmax(lesion_predictions))

    lesion_code = LESION_CLASS_CODES[lesion_index]

    lesion_class = LESION_CLASS_NAMES[lesion_code]

    lesion_confidence = float(lesion_predictions[lesion_index])

    lesion_confident = lesion_confidence >= LESION_CONFIDENCE_THRESHOLD

    # -----------------------------------------------------

    # COMMON SKIN MODEL

    # -----------------------------------------------------

    common_predictions = common_model.predict(img_array, verbose=0)[0]

    common_index = int(np.argmax(common_predictions))

    common_class = COMMON_CLASS_NAMES[common_index]

    common_confidence = float(common_predictions[common_index])

    common_confident = common_confidence >= COMMON_CONFIDENCE_THRESHOLD

    # -----------------------------------------------------

    # ONLY COMMON MODEL IS CONFIDENT

    # -----------------------------------------------------

    if common_confident and not lesion_confident:

        probabilities = {}

        for i, class_name in enumerate(COMMON_CLASS_NAMES):

            probabilities[class_name] = round(float(common_predictions[i]) * 100, 2)

        heatmap = generate_gradcam(
            common_model, common_grad_model, img_array, common_index
        )

        gradcam_image = create_gradcam_overlay(original_image, heatmap)

        return {
            "analysis_type": "unified",
            "model_used": "common_skin",
            "prediction": common_class,
            "predicted_class": common_class,
            "description": COMMON_GUIDANCE[common_class]["description"],
            "guidance": COMMON_GUIDANCE[common_class]["guidance"],
            "confidence": round(common_confidence * 100, 2),
            "uncertain": False,
            "probabilities": probabilities,
            "gradcam_image": image_to_base64(gradcam_image),
            "quality_passed": True,
            "quality_message": quality["reason"],
            "quality_details": quality,
        }

    # -----------------------------------------------------

    # ONLY LESION MODEL IS CONFIDENT

    # -----------------------------------------------------

    if lesion_confident and not common_confident:

        probabilities = {}

        for i, code in enumerate(LESION_CLASS_CODES):

            probabilities[LESION_CLASS_NAMES[code]] = round(
                float(lesion_predictions[i]) * 100, 2
            )

        heatmap = generate_gradcam(
            lesion_model, lesion_grad_model, img_array, lesion_index
        )

        gradcam_image = create_gradcam_overlay(original_image, heatmap)

        return {
            "analysis_type": "unified",
            "model_used": "lesion",
            "prediction": lesion_class,
            "predicted_code": lesion_code,
            "predicted_class": lesion_class,
            "description": LESION_GUIDANCE[lesion_code]["description"],
            "guidance": LESION_GUIDANCE[lesion_code]["guidance"],
            "confidence": round(lesion_confidence * 100, 2),
            "uncertain": False,
            "probabilities": probabilities,
            "gradcam_image": image_to_base64(gradcam_image),
            "quality_passed": True,
            "quality_message": quality["reason"],
            "quality_details": quality,
        }

    # -----------------------------------------------------

    # AMBIGUOUS / UNSUPPORTED

    # -----------------------------------------------------

    if lesion_confident and common_confident:

        reason = (
            "The image produced strong outputs "
            "from more than one specialist model. "
            "DermaWise cannot reliably select "
            "one category."
        )

    else:

        reason = (
            "The image did not produce a sufficiently "
            "strong output from the currently "
            "supported models."
        )

    return {
        "analysis_type": "unified",
        "model_used": None,
        "prediction": "Uncertain",
        "predicted_class": "Uncertain",
        "description": reason,
        "guidance": (
            "Try another clear, well-lit image. "
            "If the skin concern persists, changes, "
            "becomes painful, bleeds, or is otherwise "
            "concerning, consult a qualified "
            "healthcare professional."
        ),
        "confidence": None,
        "uncertain": True,
        "probabilities": {},
        "gradcam_image": None,
        "quality_passed": True,
        "quality_message": quality["reason"],
        "quality_details": quality,
    }


# =========================================================
# EXPERIMENTAL 114-CONDITION RESEARCH ANALYSIS
# Kept separate from /analyze because scores from independently
# trained models are not directly comparable.
# =========================================================


@app.post("/predict-broad")
async def predict_broad(file: UploadFile = File(...)):
    original_image, error = await read_uploaded_image(file)

    if error:
        return error

    quality = check_image_quality(original_image)

    if not quality["passed"]:
        return {
            "analysis_type": "broad_114_experimental",
            "quality_passed": False,
            "quality_message": quality["reason"],
            "quality_details": quality,
        }

    img_array = prepare_broad_image(original_image)

    # The KD student was trained with a logits output layer.
    logits = broad_model.predict(img_array, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()

    top_indices = np.argsort(probabilities)[::-1][:5]
    top_5 = []

    for rank, index in enumerate(top_indices, start=1):
        top_5.append(
            {
                "rank": rank,
                "class_index": int(index),
                "condition": BROAD_CLASS_NAMES[int(index)],
                "score": round(float(probabilities[index]) * 100, 2),
            }
        )

    return {
        "analysis_type": "broad_114_experimental",
        "model_used": "kd_mobilenetv3small_114",
        "prediction": top_5[0]["condition"],
        "top_5": top_5,
        "supported_categories": 114,
        "quality_passed": True,
        "quality_message": quality["reason"],
        "quality_details": quality,
        "experimental": True,
        "disclaimer": (
            "Experimental research output only. The Top-5 scores are model "
            "scores, not clinical probabilities or a medical diagnosis. "
            "This broad 114-condition model has limited classification "
            "performance and should not be used to confirm or rule out a "
            "skin condition. Seek a qualified healthcare professional for "
            "medical assessment or concerning skin changes."
        ),
    }
