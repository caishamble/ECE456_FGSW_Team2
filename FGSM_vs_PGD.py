import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
print(f"TensorFlow version: {tf.__version__}")
 
 
# ── Attack Functions ──
 
def fgsm_attack(model, x, y, epsilon):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
 
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
 
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
 
    x_adv = x + epsilon * signed_grad
    x_adv = tf.clip_by_value(x_adv, 0, 1)
 
    return x_adv
 
 
def pgd_attack(model, x, y, epsilon, alpha=0.01, iters=10):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y)
 
    x_original = tf.identity(x)
    x_adv = tf.identity(x)
 
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            prediction = model(x_adv)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
 
        gradient = tape.gradient(loss, x_adv)
        signed_grad = tf.sign(gradient)
 
        x_adv = x_adv + alpha * signed_grad
        x_adv = tf.clip_by_value(x_adv, x_original - epsilon, x_original + epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
 
    return x_adv
 
 
# ── Load MNIST ──
 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis].astype("float32")
 
print("MNIST test set loaded.")
 
 
# ── Load Model ──
# Change the path below to where your model file is located
model = tf.keras.models.load_model('my_mnist_model.keras')
y_test_oh = tf.keras.utils.to_categorical(y_test, 10)
loss, accuracy = model.evaluate(x_test, y_test_oh, verbose=1)
print(f"\nBaseline Accuracy (no attack): {accuracy * 100:.2f}%")
print(f"Baseline Loss: {loss:.4f}")
model.summary()
 
 
# ── FGSM Demo on 64 Samples ──
 
x_adv = fgsm_attack(model, x_test[:64], y_test[:64], epsilon=0.1)
 
original_preds    = tf.argmax(model(x_test[:64]), axis=1).numpy()
adversarial_preds = tf.argmax(model(x_adv), axis=1).numpy()
 
print(f"Original labels:           {y_test[:64].tolist()}")
print(f"Original predictions:      {original_preds[:64].tolist()}")
print(f"Adversarial predictions:   {adversarial_preds[:64].tolist()}")
 
 
# ── PGD Demo on 64 Samples ──
 
x_adv_pgd = pgd_attack(model, x_test[:64], y_test[:64], epsilon=0.1, alpha=0.01, iters=10)
pgd_preds = tf.argmax(model(x_adv_pgd), axis=1).numpy()
 
print(f"PGD predictions:           {pgd_preds[:64].tolist()}")
 
 
# ── Epsilon Sweep: FGSM vs PGD ──
 
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
fgsm_accuracies = []
pgd_accuracies = []
 
x_eval = x_test[:1000]
y_eval = y_test[:1000]
 
for eps in epsilons:
    x_adv_fgsm = fgsm_attack(model, x_eval, y_eval, epsilon=eps)
    fgsm_preds = tf.argmax(model(x_adv_fgsm), axis=1).numpy()
    fgsm_acc = np.mean(fgsm_preds == y_eval)
    fgsm_accuracies.append(fgsm_acc)
 
    x_adv_pgd = pgd_attack(model, x_eval, y_eval, epsilon=eps, alpha=0.01, iters=10)
    pgd_preds = tf.argmax(model(x_adv_pgd), axis=1).numpy()
    pgd_acc = np.mean(pgd_preds == y_eval)
    pgd_accuracies.append(pgd_acc)
 
    print(f"Epsilon = {eps:.2f} | FGSM Acc = {fgsm_acc:.4f} | PGD Acc = {pgd_acc:.4f}")
 
 
# ── Plot ──
 
plt.figure(figsize=(8, 5))
plt.plot(epsilons, fgsm_accuracies, marker='o', label='FGSM')
plt.plot(epsilons, pgd_accuracies, marker='s', label='PGD')
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon Curve for FGSM and PGD")
plt.legend()
plt.grid(True)
plt.savefig('chart4_fgsm_vs_pgd.png', dpi=200, bbox_inches='tight')
plt.show()
print("Chart saved as chart4_fgsm_vs_pgd.png")