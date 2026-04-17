import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
 
print(f"TensorFlow version: {tf.__version__}")
 
 
# ── FGSM Attack Function ──
 
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
 
 
# ── Quick FGSM Demo on 64 Samples ──
 
x_adv = fgsm_attack(model, x_test[:64], y_test[:64], epsilon=0.1)
 
original_preds    = tf.argmax(model(x_test[:64]), axis=1).numpy()
adversarial_preds = tf.argmax(model(x_adv), axis=1).numpy()
 
print(f"Original labels:           {y_test[:64].tolist()}")
print(f"Original predictions:      {original_preds[:64].tolist()}")
print(f"Adversarial predictions:   {adversarial_preds[:64].tolist()}")
 
 
# ── Epsilon Sweep (full test set) ──
 
epsilons = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
accuracies = []
 
BATCH = 512
 
for eps in epsilons:
    correct = 0
    total = 0
    for i in range(0, len(x_test), BATCH):
        xb = x_test[i:i+BATCH]
        yb = y_test[i:i+BATCH]
        if eps == 0.0:
            x_adv_batch = xb
        else:
            x_adv_batch = fgsm_attack(model, xb, yb, eps)
        preds = tf.argmax(model(x_adv_batch), axis=1).numpy()
        correct += np.sum(preds == yb)
        total += len(yb)
    acc = correct / total
    accuracies.append(acc)
    print(f"  e = {eps:.3f}  ->  Accuracy = {acc*100:.2f}%")
 
print("\nEpsilon sweep complete.")
 
 
# ── Chart 1: Accuracy vs Epsilon ──
 
fig, ax = plt.subplots(figsize=(8, 5))
 
ax.plot(epsilons, [a * 100 for a in accuracies],
        marker='o', markersize=7, linewidth=2.2,
        color='#E63946', markeredgecolor='white', markeredgewidth=1.5,
        zorder=3)
 
ax.axhspan(0, 50, color='#E63946', alpha=0.06)
ax.axhline(y=50, color='#E63946', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(0.42, 52, 'Below random (10-class)', fontsize=8, color='#E63946', alpha=0.7)
 
ax.set_xlabel('Perturbation Magnitude (e)', fontsize=12)
ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
ax.set_title('FGSM Attack: Model Accuracy vs Perturbation Strength',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlim(-0.01, 0.52)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
 
for i, (e, a) in enumerate(zip(epsilons, accuracies)):
    if e in [0.0, 0.1, 0.3]:
        offset = (8, 10) if a > 30 else (8, -15)
        ax.annotate(f'{a*100:.1f}%', (e, a*100),
                    textcoords='offset points', xytext=offset,
                    fontsize=9, fontweight='bold', color='#1D3557')
 
ax.grid(True, alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
plt.savefig('chart1_accuracy_vs_epsilon.png', dpi=200, bbox_inches='tight')
plt.show()
print("Chart 1 saved.")
 
 
# ── Chart 2: Per-Digit Attack Success Rate (e = 0.1) ──
 
EPS_EVAL = 0.1
 
digit_success = {}
digit_counts  = {}
 
for i in range(0, len(x_test), BATCH):
    xb = x_test[i:i+BATCH]
    yb = y_test[i:i+BATCH]
    x_adv_batch = fgsm_attack(model, xb, yb, EPS_EVAL)
    orig_preds = tf.argmax(model(xb), axis=1).numpy()
    adv_preds  = tf.argmax(model(x_adv_batch), axis=1).numpy()
 
    for digit in range(10):
        mask = (yb == digit)
        if mask.sum() == 0:
            continue
        correctly_classified = (orig_preds[mask] == digit)
        flipped = (adv_preds[mask][correctly_classified] != digit)
        digit_success[digit] = digit_success.get(digit, 0) + flipped.sum()
        digit_counts[digit]  = digit_counts.get(digit, 0)  + correctly_classified.sum()
 
success_rates = [digit_success[d] / digit_counts[d] * 100 for d in range(10)]
 
fig, ax = plt.subplots(figsize=(8, 5))
 
colors = ['#457B9D' if r < 70 else '#E63946' for r in success_rates]
bars = ax.bar(range(10), success_rates, color=colors, edgecolor='white',
              linewidth=1.2, width=0.7, zorder=3)
 
for bar, rate in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
 
ax.set_xlabel('True Digit Class', fontsize=12)
ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
ax.set_title(f'FGSM Per-Digit Vulnerability (e = {EPS_EVAL})',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(range(10))
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
 
mean_sr = np.mean(success_rates)
ax.axhline(y=mean_sr, color='#1D3557', linestyle='--', linewidth=1, alpha=0.5)
ax.text(9.4, mean_sr + 1, f'mean {mean_sr:.1f}%', fontsize=8,
        color='#1D3557', ha='right')
 
ax.grid(axis='y', alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
plt.savefig('chart2_per_digit_vulnerability.png', dpi=200, bbox_inches='tight')
plt.show()
print("Chart 2 saved.")
 
 
# ── Chart 3: Confidence Distribution: Clean vs Adversarial ──
 
EPS_CONF = 0.1
N_SAMPLES = 2000
 
x_sub = x_test[:N_SAMPLES]
y_sub = y_test[:N_SAMPLES]
 
x_adv_sub = fgsm_attack(model, x_sub, y_sub, EPS_CONF)
 
probs_clean = tf.nn.softmax(model(x_sub)).numpy()
probs_adv   = tf.nn.softmax(model(x_adv_sub)).numpy()
 
conf_clean = probs_clean[np.arange(N_SAMPLES), y_sub]
conf_adv   = probs_adv[np.arange(N_SAMPLES),   y_sub]
 
fig, ax = plt.subplots(figsize=(8, 5))
 
bins = np.linspace(0, 1, 40)
ax.hist(conf_clean, bins=bins, alpha=0.7, color='#457B9D', label='Clean inputs',
        edgecolor='white', linewidth=0.5, zorder=3)
ax.hist(conf_adv, bins=bins, alpha=0.7, color='#E63946', label='Adversarial (e=0.1)',
        edgecolor='white', linewidth=0.5, zorder=3)
 
ax.set_xlabel('Model Confidence on True Label', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Confidence Distribution: Clean vs Adversarial Inputs',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10, framealpha=0.9)
 
med_c = np.median(conf_clean)
med_a = np.median(conf_adv)
ymax = ax.get_ylim()[1]
ax.axvline(med_c, color='#457B9D', linestyle='--', linewidth=1.2)
ax.axvline(med_a, color='#E63946', linestyle='--', linewidth=1.2)
ax.text(med_c + 0.02, ymax * 0.9, f'median {med_c:.2f}', fontsize=8, color='#457B9D')
ax.text(med_a + 0.02, ymax * 0.82, f'median {med_a:.2f}', fontsize=8, color='#E63946')
 
ax.grid(True, alpha=0.25)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
plt.savefig('chart3_confidence_distribution.png', dpi=200, bbox_inches='tight')
plt.show()
print("Chart 3 saved.")
 
 
print("\nAll 3 charts generated and saved as PNG files.")