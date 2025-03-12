import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model

# Create a new MobileNetV2 model to replace the corrupted model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(4, activation='softmax')(x)

# Create the new fixed model
model = Model(inputs=base_model.input, outputs=x)

# Save the fixed model
fixed_file_path = 'fixed_rice.h5'
save_model(model, fixed_file_path)

print("✅ A brand new model 'fixed_rice.h5' has been created successfully.")
