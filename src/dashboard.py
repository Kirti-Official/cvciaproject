import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from torchvision import transforms

from cbam_model import EfficientNetCBAM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Screening",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_b0_cbam_best.pth"

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = EfficientNetCBAM(
        num_classes=5
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # Convert RGB to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # Detect retinal region
    # --------------------------------------------------------

    mask = gray > 10

    if np.any(mask):

        coords = np.argwhere(mask)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        padding = 5

        y_min = max(
            0,
            y_min - padding
        )

        x_min = max(
            0,
            x_min - padding
        )

        y_max = min(
            image.shape[0] - 1,
            y_max + padding
        )

        x_max = min(
            image.shape[1] - 1,
            x_max + padding
        )

        image = image[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------

    target_size = 224

    h, w = image.shape[:2]

    scale = min(
        target_size / w,
        target_size / h
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Create 224 x 224 canvas
    # --------------------------------------------------------

    canvas = np.zeros(
        (target_size, target_size, 3),
        dtype=np.uint8
    )

    x_offset = (
        target_size - new_w
    ) // 2

    y_offset = (
        target_size - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        canvas,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        [
            l_channel,
            a_channel,
            b_channel
        ]
    )

    processed = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return processed


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    input_tensor,
    predicted_class,
    image
):

    activations = None
    gradients = None

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(
        module,
        inputs,
        output
    ):
        nonlocal activations
        activations = output

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        nonlocal gradients
        gradients = grad_output[0]

    # --------------------------------------------------------
    # Target CBAM layer
    # --------------------------------------------------------

    target_layer = model.cbam

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        model.zero_grad()

        output = model(
            input_tensor
        )

        # ----------------------------------------------------
        # Select predicted class
        # ----------------------------------------------------

        target = output[
            0,
            predicted_class
        ]

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        target.backward()

        # ----------------------------------------------------
        # Check hooks
        # ----------------------------------------------------

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Calculate channel weights
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted activation map
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1
        ).squeeze()

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = F.relu(
            cam
        )

        cam = cam.detach().cpu().numpy()

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam = cam - cam.min()

        if cam.max() > 0:

            cam = cam / cam.max()

        # ----------------------------------------------------
        # Resize CAM to image size
        # ----------------------------------------------------

        cam = cv2.resize(
            cam,
            (
                image.shape[1],
                image.shape[0]
            )
        )

        # ----------------------------------------------------
        # Convert CAM to heatmap
        # ----------------------------------------------------

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Create overlay
        # ----------------------------------------------------

        overlay = (
            0.55 * image +
            0.45 * heatmap
        )

        overlay = np.clip(
            overlay,
            0,
            255
        ).astype(
            np.uint8
        )

        return heatmap, overlay

    finally:

        # ----------------------------------------------------
        # Remove hooks
        # ----------------------------------------------------

        forward_handle.remove()
        backward_handle.remove()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🩺 AI-Based Diabetic Retinopathy Screening"
)

st.markdown(
    """
    **Deep Learning for Diabetic Retinopathy Screening
    from Retinal Fundus Images**

    **EfficientNet-B0 + CBAM + Grad-CAM**
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ System Information"
)

st.sidebar.write(
    "**Dataset:** APTOS 2019"
)

st.sidebar.write(
    "**Model:** EfficientNet-B0"
)

st.sidebar.write(
    "**Attention:** CBAM"
)

st.sidebar.write(
    "**Explainability:** Grad-CAM"
)

st.sidebar.write(
    "**Input Size:** 224 × 224"
)

st.sidebar.write(
    "**Classes:** 5"
)

st.sidebar.write(
    "**Device:** CPU"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header(
    "📤 Upload Retinal Fundus Image"
)

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header(
    "📊 Model Performance"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Validation Accuracy",
        "78.99%"
    )

with col2:

    st.metric(
        "Macro F1",
        "60.52%"
    )

with col3:

    st.metric(
        "Quadratic Weighted Kappa",
        "85.46%"
    )


# ============================================================
# SCREENING PIPELINE
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # READ UPLOADED IMAGE
    # ========================================================

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Unable to read the uploaded image."
        )

        st.stop()

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    processed_image = preprocess_image(
        image
    )

    # ========================================================
    # DISPLAY ORIGINAL + PREPROCESSED
    # ========================================================

    st.divider()

    st.header(
        "🖼️ Image Processing"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Preprocessed Image"
        )

        st.image(
            processed_image,
            use_container_width=True
        )

    # ========================================================
    # PREPARE MODEL INPUT
    # ========================================================

    input_tensor = transform(
        processed_image
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    input_tensor = input_tensor.to(
        DEVICE
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # PREDICTION
    # ========================================================

    output = model(
        input_tensor
    )

    probabilities = F.softmax(
        output,
        dim=1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_class
    ].item()

    # ========================================================
    # SCREENING RESULT
    # ========================================================

    st.divider()

    st.header(
        "🔍 Screening Result"
    )

    result_col1, result_col2 = st.columns(
        2
    )

    with result_col1:

        st.success(
            f"Predicted Stage: "
            f"{CLASS_NAMES[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with result_col2:

        st.subheader(
            "📊 Class Probabilities"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                probabilities[i].item()
            )

            st.write(
                f"**{class_name}**"
            )

            st.progress(
                probability,
                text=f"{probability * 100:.2f}%"
            )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()

    st.header(
        "🧠 Explainable AI — Grad-CAM"
    )

    st.markdown(
        """
        Grad-CAM highlights image regions that contributed
        to the model's prediction.
        """
    )

    try:

        heatmap, overlay = generate_gradcam(
            model,
            input_tensor,
            predicted_class,
            processed_image
        )

        # ----------------------------------------------------
        # Display Grad-CAM results
        # ----------------------------------------------------

        cam_col1, cam_col2, cam_col3 = st.columns(
            3
        )

        with cam_col1:

            st.subheader(
                "Original"
            )

            st.image(
                processed_image,
                use_container_width=True
            )

        with cam_col2:

            st.subheader(
                "Grad-CAM Heatmap"
            )

            st.image(
                heatmap,
                use_container_width=True
            )

        with cam_col3:

            st.subheader(
                "Attention Overlay"
            )

            st.image(
                overlay,
                use_container_width=True
            )

        st.info(
            "Grad-CAM indicates image regions that "
            "contributed to the model's prediction. "
            "It is an explainability visualization, "
            "not a direct lesion detector."
        )

    except Exception as e:

        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Computer Vision Project | "
    "EfficientNet-B0 + CBAM | "
    "APTOS 2019"
)