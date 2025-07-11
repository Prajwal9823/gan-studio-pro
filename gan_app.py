import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Configuration
st.set_page_config(layout="wide", page_title="Generative Adversarial Networks For Image Generation", page_icon="🤖")
st.sidebar.title("Control Panel")

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        if not os.path.exists('generator_model.h5'):
            st.error("Generator model not found. Please train the model first.")
            return None, None

        generator = tf.keras.models.load_model('generator_model.h5', compile=False)
        classifier_path = 'mnist_classifier.h5'

        if os.path.exists(classifier_path):
            classifier = tf.keras.models.load_model(classifier_path)
        else:
            with st.spinner("Training classifier..."):
                (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
                x_train = np.expand_dims(x_train, -1) / 255.0

                classifier = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])


                classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                classifier.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
                classifier.save(classifier_path)

        return generator, classifier
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

generator, classifier = load_models()

def generate_images(noise):
    if generator is None:
        return None
    try:
        images = generator(noise, training=False)
        return (images.numpy() * 127.5 + 127.5).astype(np.uint8)
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None

def main():
    with st.sidebar:
        st.header("🎛️ Generator Settings")
        num_images = st.slider("Number of images", 1, 16, 9)
        noise_seed = st.number_input("Noise seed", 0, 1000, 42)
        generate_btn = st.button("Generate Images")

        st.markdown("---")
        st.header("🖌️ Drawing Canvas")
        stroke_width = st.slider("Brush Size", 1, 30, 15)
        stroke_color = st.color_picker("Stroke Color", "#000000")
        bg_color = st.color_picker("Background Color", "#FFFFFF")
        drawing_mode = st.selectbox("Drawing Mode", ["freedraw", "line", "rect", "circle"])
        clear_canvas = st.button("Clear Canvas")

        st.markdown("---")
        st.header("📊 Training Info")
        st.success("Generator loaded" if generator else "Generator not found")
        st.success("Classifier loaded" if classifier else "Classifier not loaded")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("🤖 Generated Images")

        if generate_btn and generator:
            tf.random.set_seed(noise_seed)
            noise = tf.random.normal([num_images, 100])
            generated_images = generate_images(noise)

            if generated_images is not None:
                cols = 3
                rows = (num_images + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3))
                axes = axes.flatten()

                for i in range(num_images):
                    axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
                    axes[i].axis('off')

                for j in range(num_images, len(axes)):
                    axes[j].axis('off')

                st.pyplot(fig)

        st.header("📈 Training Progression")
        epoch_files = sorted([f for f in os.listdir() if f.startswith('image_at_epoch_') and f.endswith('.png')])
        if epoch_files:
            cols = st.columns(5)
            for i, f in enumerate(epoch_files[:5]):
                with cols[i % 5]:
                    st.image(f, caption=f.replace('.png', '').replace('_', ' '))
        else:
            st.warning("No training images found")

    with col2:
        st.header("✏️ Drawing Canvas")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=400,
            width=400,
            drawing_mode=drawing_mode,
            update_streamlit=True,
            key="drawing_canvas",
        )

        if clear_canvas:
            canvas_result = None
            st.experimental_rerun()

        if canvas_result is not None and canvas_result.image_data is not None:
            img_pil = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            st.subheader("Your Drawing")
            st.image(img_pil, width=300)

           
            img = img_pil.convert("L")
            img = 255 - np.array(img)  

            # Crop to bounding box
            coords = np.column_stack(np.where(img > 30))
            if coords.size == 0:
                st.warning("Please draw a digit.")
                return
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img = img[y0:y1 + 1, x0:x1 + 1]

            # Resize and pad
            img_pil = Image.fromarray(img).resize((20, 20), Image.Resampling.LANCZOS)
            img_pil = ImageOps.invert(img_pil)  # Back to white digit on black
            final_img = Image.new('L', (28, 28), 255)
            final_img.paste(img_pil, (4, 4))

            # Display processed
            st.subheader("Processed for Classifier")
            st.image(final_img, width=150, clamp=True)

            # Predict
            img_array = np.array(final_img)
            img_array = 255 - img_array  # Back to black digit on white
            img_array = img_array / 255.0
            img_input = img_array.reshape(1, 28, 28, 1)

            if classifier:
                try:
                    pred = classifier.predict(img_input, verbose=0)[0]
                    pred = np.clip(pred, 0, 1)
                    pred = pred / pred.sum() if pred.sum() > 0 else np.ones_like(pred) / len(pred)

                    st.subheader("Classifier Prediction")
                    for i, prob in enumerate(pred):
                        display_prob = max(0.0, min(1.0, float(prob)))
                        st.progress(display_prob, text=f"{i}: {prob:.1%}")
                except Exception as e:
                    st.error(f"Classification failed: {str(e)}")

        if classifier and generator:
            st.header("🔍 Classifier Feedback")
            if st.button("Test Random Generation"):
                noise = tf.random.normal([1, 100])
                img = generate_images(noise)

                if img is not None:
                    input_img = img[0, :, :, 0]
                    input_img = np.expand_dims(input_img, axis=(0, -1)) / 255.0

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(input_img[0, :, :, 0], width=150)
                    with col2:
                        try:
                            pred = classifier.predict(input_img, verbose=0)[0]
                            pred = np.clip(pred, 0, 1)
                            pred = pred / pred.sum() if pred.sum() > 0 else np.ones_like(pred) / len(pred)
                            for i, prob in enumerate(pred):
                                st.progress(float(prob), text=f"{i}: {float(prob):.1%}")
                        except Exception as e:
                            st.error(f"Classification failed: {str(e)}")

if __name__ == "__main__":
    main()
