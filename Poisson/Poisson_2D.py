import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)


# Define the B-spline representation for the 2D Poisson problem
class bsplines:

    def __init__(self, n_basis, degree):

        self.degree = degree
        self.n_basis = n_basis
        self.n_knots = n_basis + degree + 1

        # Uniform clamped bsplines
        knot_vector = torch.linspace(0, 1, self.n_knots - 2 * degree)
        self.knot_vector = torch.cat((
                torch.full((degree,), knot_vector[0]),
                knot_vector,
                torch.full((degree,), knot_vector[-1])
            ))
        
        # Same knot vector in every direction
        self.knot_x = self.knot_vector.clone()
        self.knot_y = self.knot_vector.clone()

        # 2D trainable coefficients
        self.coeffs = torch.rand(n_basis, n_basis, requires_grad=True)

        
    def basis_matrix(self, x, degree=None, knots=None):

        """Evaluate all B-spline basis functions without autograd."""
        degree = self.degree if degree is None else degree
        knots = self.knot_vector if knots is None else knots
        knots = knots.to(device=x.device, dtype=x.dtype)
        B = ((x[:, None] >= knots[:-1]) &
             (x[:, None] < knots[1:])).to(x.dtype)
        for p in range(1, degree + 1):
            n = len(knots) - p - 1
            next_B = torch.zeros((x.numel(), n), device=x.device, dtype=x.dtype)
            for i in range(n):
                denom1 = knots[i + p] - knots[i]
                denom2 = knots[i + p + 1] - knots[i + 1]
                if denom1 > 0:
                    next_B[:, i] += (x - knots[i]) / denom1 * B[:, i]
                if denom2 > 0:
                    next_B[:, i] += (knots[i + p + 1] - x) / denom2 * B[:, i + 1]
            B = next_B
        at_right_endpoint = x == knots[-1]
        if torch.any(at_right_endpoint):
            B[at_right_endpoint] = 0
            B[at_right_endpoint, -1] = 1
        return B

    def basis_and_second_derivative(self, x, knots=None):
        """Return B_{i,p}(x) and its analytic second derivative."""
        knots = self.knot_vector if knots is None else knots
        knots = knots.to(device=x.device, dtype=x.dtype)
        p = self.degree
        if p < 2:
            raise ValueError('Second derivatives require degree >= 2')
        B = self.basis_matrix(x, degree=p, knots=knots)
        B_low = self.basis_matrix(x, degree=p - 2, knots=knots)
        D_low = torch.zeros((x.numel(), self.n_basis + 1),
                            device=x.device, dtype=x.dtype)
        q = p - 1
        for i in range(self.n_basis + 1):
            denom1 = knots[i + q] - knots[i]
            denom2 = knots[i + q + 1] - knots[i + 1]
            if denom1 > 0:
                D_low[:, i] += q / denom1 * B_low[:, i]
            if denom2 > 0:
                D_low[:, i] -= q / denom2 * B_low[:, i + 1]
        D2 = torch.zeros_like(B)
        for i in range(self.n_basis):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            if denom1 > 0:
                D2[:, i] += p / denom1 * D_low[:, i]
            if denom2 > 0:
                D2[:, i] -= p / denom2 * D_low[:, i + 1]
        return B, D2

    def forward(self, x):
        B_x = self.basis_matrix(x[:, 0], knots=self.knot_x)[:, 1:-1]
        B_y = self.basis_matrix(x[:, 1], knots=self.knot_y)[:, 1:-1]
        return torch.einsum('ni,ij,nj->n', B_x, self.coeffs[1:-1, 1:-1], B_y)


# Define the B-spline model with n= basis functions and degree p=3
model = bsplines(20,3)

# Define the collocation points in the interior of the domain
colloc_points = 5000
X = torch.rand(colloc_points,2)







# Right-hand side of Poisson equation
def f(x, y):
    return -2 * (np.pi ** 2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

# Precompute the fixed B-spline Laplacian collocation operator once.
with torch.no_grad():
    Bx, Bxx = model.basis_and_second_derivative(X[:, 0], model.knot_x)
    By, Byy = model.basis_and_second_derivative(X[:, 1], model.knot_y)
    Bx, Bxx = Bx[:, 1:-1], Bxx[:, 1:-1]
    By, Byy = By[:, 1:-1], Byy[:, 1:-1]
    laplacian_matrix = (
        Bxx[:, :, None] * By[:, None, :]
        + Bx[:, :, None] * Byy[:, None, :]
    ).reshape(X.shape[0], -1)
    source = f(X[:, 0], X[:, 1])



#### Taining loop###
loss_vec = []

epochs = 20000
learning_rate=1e-3
optimizer = torch.optim.Adam([model.coeffs], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=50, verbose=True, min_lr=1e-5
)

# Early stopping monitors the fixed collocation residual. Its patience is
# intentionally longer than the scheduler patience, allowing several LR drops.
early_stopping_patience = 500
min_delta = 1e-8
best_loss = float('inf')
best_coeffs = model.coeffs.detach().clone()
patience_counter = 0



for epoch in range(epochs):
    optimizer.zero_grad()
    interior_coeffs = model.coeffs[1:-1, 1:-1].reshape(-1)
    res = laplacian_matrix @ interior_coeffs - source
    
    loss = torch.mean(res**2)
    
    loss.backward()
    ###########Changes#################
    optimizer.step()

    loss_value = loss.item()
    scheduler.step(loss_value)

    # A decrease must exceed min_delta to count as an improvement.
    if loss_value < best_loss - min_delta:
        best_loss = loss_value
        best_coeffs = model.coeffs.detach().clone()
        patience_counter = 0
    else:
        patience_counter += 1
    

    loss_vec.append(loss_value)
    # Print loss and parameters every 100 iterations
    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_value}, LR: {current_lr:.2e}')

    if patience_counter >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch + 1}; best loss: {best_loss:.6e}')
        break

# Continue with the best coefficients rather than the final stagnant iterate.
with torch.no_grad():
    model.coeffs.copy_(best_coeffs)




# Plot the solution versus the exact solution
x = y = np.linspace(0, 1, 100)
Xg, Yg = np.meshgrid(x, y)
XY = np.vstack([Xg.ravel(), Yg.ravel()]).T
XY_torch = torch.tensor(XY, dtype=torch.float32)
with torch.no_grad():
    u_pred = model.forward(XY_torch).numpy().reshape(Xg.shape)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

u_exact = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)

# PINN solution
im0 = axs[0].imshow(u_pred, extent=[0,1,0,1], origin='lower', cmap='viridis')
axs[0].set_title("PI-TP-BsN Solution",fontsize=14)
cbar = fig.colorbar(im0, ax=axs[0])
cbar.ax.tick_params(labelsize=12) 

# Exact solution
im1 = axs[1].imshow(u_exact, extent=[0,1,0,1], origin='lower', cmap='viridis')
axs[1].set_title("Exact Solution",fontsize=14)
cbar = fig.colorbar(im1, ax=axs[1])
cbar.ax.tick_params(labelsize=12) 

# Absolute error
error = np.abs(u_pred - u_exact)
im2 = axs[2].imshow(error, extent=[0,1,0,1], origin='lower', cmap='hot')
axs[2].set_title("Absolute Error",fontsize=14)
cbar = fig.colorbar(im2, ax=axs[2])
cbar.ax.tick_params(labelsize=12) 



for ax in axs:
    ax.set_xlabel("x",fontsize=14)
    ax.set_ylabel("y",fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()


#Plot the loss curve
loss_vec_npy = np.array(loss_vec)
plt.plot(loss_vec_npy)
plt.yscale('log')
plt.title("Loss Curve",fontsize=14)
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Loss",fontsize=14)
plt.show()