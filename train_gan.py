import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import struct

def load_mnist(path):
    """Load MNIST from ubyte files"""
    images = []
    for kind in ['train', 't10k']:
        img_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
        with open(img_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images.append(np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols))
    return np.concatenate(images)

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 100
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

# Load dataset
print("Loading dataset...")
all_images = load_mnist('data')
print(f"Loaded {len(all_images)} images")

# Preprocess data
all_images = all_images.reshape(all_images.shape[0], 28, 28, 1).astype('float32')
all_images = (all_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices(all_images)
train_dataset = train_dataset.shuffle(70000).batch(BATCH_SIZE)

# Generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7, 7, 256)))
    
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    
    return model

# Discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', 
                             input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Initialize models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        
        # Print losses
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}')
        
        # Generate sample images
        if epoch % 20 == 0 or epoch == epochs - 1:
            generate_and_save_images(generator, epoch + 1)
    
    # Save generator
    generator.save('generator_model.h5')
    print("Training completed! Generator saved as generator_model.h5")

# Generate and save images
def generate_and_save_images(model, epoch):
    predictions = model(tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM]), training=False)
    fig = plt.figure(figsize=(4,4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()

# Start training
print("Starting training...")
train(train_dataset, EPOCHS)