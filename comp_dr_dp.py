import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import modules
from utils.modules import SingleBVPNet

# === Settings ===
model_path = 'runs/rel2car_dp_match_value_var_4/training/checkpoints/model_epoch_100000.pth'
dp_path = 'op_dp/rel2car_dp.npy'
theta_val = 0.0
t_val = 0.0
grid_min, grid_max, grid_res = -4, 4, 100


# === Load DP Value Function ===
print("Loading Optimized DP result...")
dp_value = np.load(dp_path)  # shape (Nx, Ny, Nθ)
dp_slice = dp_value[:, :, dp_value.shape[2] // 2]  # θ = 0 slice
print("DP value range:", dp_slice.min(), "to", dp_slice.max())

# === Create Grid ===
x = np.linspace(grid_min, grid_max, dp_slice.shape[0])
y = np.linspace(grid_min, grid_max, dp_slice.shape[1])
xx, yy = np.meshgrid(x, y)
input_tensor = torch.tensor(np.stack([
    xx.ravel(), yy.ravel(),
    np.full_like(xx.ravel(), theta_val),
    np.full_like(xx.ravel(), t_val)
], axis=1), dtype=torch.float32)

# === Load DeepReach model ===
print("Loading DeepReach model...")
model = SingleBVPNet(
    in_features=4,
    out_features=1,
    type='sine',
    mode='mlp',
    hidden_features=512,
    num_hidden_layers=3
)
checkpoint = torch.load(model_path)
print("Checkpoint keys:", checkpoint.keys())
model.load_state_dict(checkpoint['model'])
model.eval()

# === Evaluate DeepReach ===
print("Evaluating DeepReach model...")
value_var = 0.5
value_mean = 1.0
with torch.no_grad():
    model_input = {'coords': input_tensor}
    output = model(model_input)
    V_deepreach = output['model_out'].reshape(xx.shape).numpy()
    V_actual = V_deepreach * 4

print("DeepReach value range:", V_deepreach.min(), "to", V_deepreach.max())
print("V_actual value range:", V_actual.min(), "to", V_actual.max())


# === Plotting ===
plt.figure(figsize=(10, 6))
cont1 = plt.contour(xx, yy, V_actual, levels=[0.0], colors='red', label='DeepReach')
cont2 = plt.contour(xx, yy, dp_slice, levels=[0.0], colors='blue', label='DP')


plt.title("BRT Comparison: DeepReach (red) vs Optimized DP (blue)")
plt.xlabel("x_r")
plt.ylabel("y_r")
plt.grid(True)
plt.legend([cont1.collections[0], cont2.collections[0]], ['DeepReach', 'Optimized DP'])
plt.axis('equal')
plt.tight_layout()
plt.savefig("brt_comparison.png")
plt.show()

# === Compute Metrics ===
mae = np.mean(np.abs(V_deepreach - dp_slice))
mask_deepreach = V_deepreach <= 0
mask_dp = dp_slice <= 0
intersection = np.logical_and(mask_deepreach, mask_dp).sum()
union = np.logical_or(mask_deepreach, mask_dp).sum()
iou = intersection / union

print(f"\n📊 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📐 Intersection over Union (IoU): {iou:.4f}")
