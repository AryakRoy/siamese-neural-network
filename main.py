import cv2
import os
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


def capture_images():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[170:170+250, 530:530+250, :]
        cv2.imshow('Image Collection', frame)

        # capture anchor image
        if cv2.waitKey(1) & 0XFF == ord('a'):
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        # capture positive image
        if cv2.waitKey(1) & 0XFF == ord('p'):
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        # breaking key
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH, "*.jpg")).take(300)
positive = tf.data.Dataset.list_files(
    os.path.join(POS_PATH, "*.jpg")).take(300)
negative = tf.data.Dataset.list_files(
    os.path.join(NEG_PATH, "*.jpg")).take(300)

dir_test = positive.as_numpy_iterator()


def pre_process(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


positives = tf.data.Dataset.zip(
    (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip(
    (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


def preprocess_twin(input_img, validation_img, label):
    return(pre_process(input_img), pre_process(validation_img), label)


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Model Architecture

# Embedding Layer


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_embedding()

print(embedding.summary())

# Distance Layer


class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    # similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Siamese Model


def make_siamese_model():

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),
                              embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()
print(siamese_model.summary())

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)

# Custom Training Step


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        ypred = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, ypred)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


def train(data, EPOCHS):
    # Loop through epochs
    # add training loss array
    # add training accuracy array
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            progbar.update(idx+1)
            # calculate training accuracy

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


EPOCHS = 100

# train model
train(train_data, EPOCHS)

plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# Model Evaluation
Y_true = np.array([])
Y_pred = np.array([])
iterator = test_data.as_numpy_iterator()
num_batch = 0
for batch in test_data:
    num_batch += 1
print("Num Batch: ", num_batch)
for i in range(num_batch):
    try:
        test_input, test_val, y_true = iterator.next()
        y_pred = model.predict([test_input, test_val])
        y_pred = [1 if prediction > 0.5 else 0 for prediction in y_pred]
        Y_pred = np.append(Y_pred, y_pred)
        Y_true = np.append(Y_true, y_true)
    except tf.errors.OutOfRangeError:
        break

c = confusion_matrix(Y_true, Y_pred).numpy()
accuracy = (c[0, 0] + c[1, 1]) / (c[0, 0] + c[1, 1] + c[0, 1] + c[1, 0])
print(accuracy*100)

# Model Verification
for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join(
        'application_data', 'verification_images', image)
    print(validation_img)


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = pre_process(os.path.join(
            'application_data', 'input_image', 'input_image.jpeg'))
        validation_img = pre_process(os.path.join(
            'application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(
            list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) >= detection_threshold)
    print(f"Detection : {detection}")

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / \
        len(os.listdir(os.path.join('application_data', 'verification_images')))
    print(f"Verification : {verification}")
    verified = verification > verification_threshold

    return results, verified


def verification_capture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[170:170+250, 530:530+250, :]

        cv2.imshow('Verification', frame)

        # Verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):
            # Save input image to application_data/input_image folder
            cv2.imwrite(os.path.join('application_data/',
                                     'input_image', 'input_image.jpeg'), frame)
            # Run verification
            results, verified = verify(model, 0.3, 0.5)
            print(verified)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
