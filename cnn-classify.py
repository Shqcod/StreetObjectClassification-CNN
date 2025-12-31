# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
IMG_SIZE= (224, 224)
N_CLASSES= 7

# %%
df= pd.read_csv('paths.csv')

# %%
df['image'] = df['image'].str.replace(
    '/kaggle/input/street-objects/', 
    '', 
    regex=False
)

# %%
df['label'].unique()

# %%
df['label']= df['label'].astype('string')

# %%
from sklearn.model_selection import train_test_split

# %%
df_trn, df_tmp= train_test_split(df,     test_size=0.3, random_state=42, shuffle=True, stratify=df['label'])
df_vld, df_tst= train_test_split(df_tmp, test_size=0.4, random_state=42, shuffle=True, stratify=df_tmp['label'])

print(f"Training:   {len(df_trn):4d} images")
print(f"Validation: {len(df_vld):4d} images")
print(f"Test:       {len(df_tst):4d} images")

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models              import Sequential
from tensorflow.keras.layers              import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers          import Adamax
from tensorflow.keras.callbacks		      import EarlyStopping

# %%
BATCH_SIZE= 16
EPOCHS= 10
LEARN_RATE= 1e-3

# %%
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# %%
def preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE) 
    img = img / 255.0  
    return img, label

# %%
def create_tf_dataset(df, batch_size):
    paths = df['image'].values  
    labels = df['label'].values
    labels = to_categorical(labels, num_classes=N_CLASSES)  
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(df)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# %%
train_dataset = create_tf_dataset(df_trn, BATCH_SIZE)
validation_dataset = create_tf_dataset(df_vld, BATCH_SIZE)
test_dataset = create_tf_dataset(df_tst, BATCH_SIZE)

# %%
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# %%
model = create_cnn_model((IMG_SIZE[0], IMG_SIZE[1], 3), N_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)

# %%
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")

# %%
predictions = model.predict(test_dataset)

predicted_classes = tf.argmax(predictions, axis=1)

actual_classes = tf.concat([y for x, y in test_dataset], axis=0)
actual_classes = tf.argmax(actual_classes, axis=1)

print(f"Predicted: {predicted_classes.numpy()}")
print(f"Actual: {actual_classes.numpy()}")


# %%
def visualize_predictions_with_images(test_dataset, model, num_images=5):
    # Create an iterator for the dataset
    dataset_iter = iter(test_dataset)

    # Display 'num_images' sets of images
    for i in range(num_images):
        # Get a batch of images and labels
        image_batch, label_batch = next(dataset_iter)

        # Predict the labels using the trained model
        predictions = model.predict(image_batch)

        # Assuming you want to plot a maximum of 5 images from a batch
        num_to_display = min(5, image_batch.shape[0])  # Handle less than 5 images if batch size is smaller

        plt.figure(figsize=(15, 15))  # Adjust the figure size based on your layout preference
        for j in range(num_to_display):
            plt.subplot(1, num_to_display, j + 1)  # Adjust subplot layout based on number of images
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image_batch[j])
            predicted_label = np.argmax(predictions[j])
            actual_label = np.argmax(label_batch[j])
            plt.title(f"Pred: {predicted_label}, Actual: {actual_label}")

        plt.show()

# Example usage
visualize_predictions_with_images(test_dataset, model, num_images=1)


