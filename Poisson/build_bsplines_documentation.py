"""Build the mathematical documentation for Poisson/Bspline_2D.py as a PDF."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUTPUT_PATH = Path(__file__).with_name("Bspline_2D_documentation.pdf")
PAGE_SIZE = (8.27, 11.69)  # A4 in inches


def _wrapped_lines(text, width=94):
    """Wrap prose while retaining explicit paragraph breaks."""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
        else:
            lines.extend(
                textwrap.wrap(
                    paragraph,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return lines


def _add_page(pdf, title, blocks, page_number):
    """Render one page from heading, prose, equation, and code blocks."""
    figure = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    figure.subplots_adjust(0, 0, 1, 1)
    figure.text(
        0.08,
        0.955,
        title,
        fontsize=17,
        fontweight="bold",
        color="#183153",
        va="top",
    )
    figure.lines.append(
        plt.Line2D(
            [0.08, 0.92],
            [0.928, 0.928],
            transform=figure.transFigure,
            color="#4f81bd",
            linewidth=1.2,
        )
    )

    y = 0.895
    for kind, content in blocks:
        if kind == "heading":
            y -= 0.008
            figure.text(
                0.08,
                y,
                content,
                fontsize=12.2,
                fontweight="bold",
                color="#244b74",
                va="top",
            )
            y -= 0.038
        elif kind == "prose":
            for line in _wrapped_lines(content):
                figure.text(
                    0.085,
                    y,
                    line,
                    fontsize=9.5,
                    color="#202020",
                    va="top",
                    family="sans-serif",
                )
                y -= 0.022
            y -= 0.010
        elif kind == "equation":
            figure.text(
                0.12,
                y,
                content,
                fontsize=12,
                color="#111111",
                va="top",
            )
            y -= 0.047
        elif kind == "code":
            for line in content.split("\n"):
                figure.text(
                    0.10,
                    y,
                    line,
                    fontsize=8.7,
                    color="#7a1f1f",
                    va="top",
                    family="monospace",
                )
                y -= 0.021
            y -= 0.010
        elif kind == "note":
            for line in _wrapped_lines(content, width=88):
                figure.text(
                    0.10,
                    y,
                    line,
                    fontsize=9.2,
                    color="#3d5266",
                    va="top",
                    style="italic",
                )
                y -= 0.022
            y -= 0.010
        else:
            raise ValueError(f"Unknown block kind: {kind}")

    figure.text(
        0.5,
        0.035,
        f"Poisson/Bspline_2D.py mathematical documentation  |  {page_number}",
        fontsize=8,
        color="#777777",
        ha="center",
    )
    pdf.savefig(figure, bbox_inches=None)
    plt.close(figure)


PAGES = [
    (
        "B-spline solution of the two-dimensional Poisson equation",
        [
            (
                "prose",
                "Mathematical documentation for Poisson/Bspline_2D.py. The program "
                "approximates a known Poisson problem on the unit square with a "
                "tensor-product B-spline surface. Spatial derivatives of the fixed "
                "basis functions are evaluated analytically and assembled before "
                "optimization.",
            ),
            ("heading", "Problem solved"),
            ("equation", r"$\Omega=(0,1)\times(0,1),\qquad \Delta u=f\quad\mathrm{in}\ \Omega,$"),
            ("equation", r"$u=0\quad\mathrm{on}\ \partial\Omega,$"),
            ("equation", r"$f(x,y)=-2\pi^2\sin(\pi x)\sin(\pi y).$"),
            (
                "prose",
                "The manufactured exact solution is u_exact(x,y) = sin(pi x) sin(pi y), "
                "because differentiating it twice in x and y gives the prescribed source. "
                "The exact field is used only for error measurement and visualization; "
                "it is not used to train the coefficients.",
            ),
            ("heading", "Approximation"),
            ("equation", r"$u_h(x,y)=\sum_{i=0}^{n_x-1}\sum_{j=0}^{n_y-1} C_{ij}N_{i,p}(x)M_{j,p}(y).$"),
            (
                "prose",
                "N and M are one-dimensional B-spline bases, p is the common degree, "
                "and C is the trainable coefficient matrix. In this script n_x = n_y "
                "= n_basis. Flattening the interior entries of C gives the optimization "
                "vector c.",
            ),
            (
                "note",
                "Important: at one point (x,y), the basis outer product is a matrix. "
                "At N collocation points, stacking those matrices produces a third-order "
                "tensor before it is flattened into the design matrix.",
            ),
        ],
    ),
    (
        "1. Class bsplines: construction and knot vector",
        [
            ("heading", "bsplines.__init__(n_basis, degree, device)"),
            (
                "prose",
                "The constructor validates the spline space, creates an open clamped "
                "uniform knot vector, and allocates the coefficient matrix C as a "
                "PyTorch parameter. A degree-p space needs at least p+1 basis functions.",
            ),
            ("equation", r"$n_{\mathrm{knots}}=n_{\mathrm{basis}}+p+1.$"),
            (
                "prose",
                "For an open knot vector on [0,1], both endpoints occur p+1 times. "
                "There are n_basis-p-1 uniformly spaced internal knots:",
            ),
            ("equation", r"$T=(0_0,\ldots,0_p,\ \xi_1,\ldots,\xi_m,\ 1_0,\ldots,1_p).$"),
            (
                "prose",
                "Clamping makes the first basis function equal to one at x=0 and the "
                "last basis function equal to one at x=1. This endpoint interpolation "
                "property is what permits boundary coefficients to be fixed exactly.",
            ),
            ("heading", "Coefficient dimensions"),
            ("equation", r"$C\in\mathbb{R}^{n_{\mathrm{basis}}\times n_{\mathrm{basis}}}.$"),
            (
                "prose",
                "C is initialized with small random values. Although the full matrix is "
                "stored, forward evaluation uses C[1:-1,1:-1]. Therefore all coefficients "
                "touching the four sides are implicitly zero and do not participate in "
                "the solution.",
            ),
            ("equation", r"$C_{0j}=C_{n_x-1,j}=C_{i0}=C_{i,n_y-1}=0.$"),
            (
                "code",
                "Constructor output:\n"
                "  self.knots : one-dimensional knot vector\n"
                "  self.C     : trainable n_basis x n_basis tensor",
            ),
        ],
    ),
    (
        "2. basis_matrix(x): Cox-de Boor evaluation",
        [
            ("heading", "Degree-zero basis"),
            (
                "prose",
                "For every coordinate x_q, basis_matrix first forms the piecewise "
                "constant functions. Each is the indicator of one knot span:",
            ),
            ("equation", r"$N_{i,0}(x)=1\quad\mathrm{if}\quad T_i\leq x<T_{i+1};\qquad N_{i,0}(x)=0\quad\mathrm{otherwise}.$"),
            (
                "prose",
                "The right endpoint x=1 is treated separately so that it belongs to the "
                "last basis function. Without this explicit convention all half-open "
                "interval tests would evaluate to zero there.",
            ),
            ("heading", "Recursive elevation to degree p"),
            ("equation", r"$N_{i,k}(x)=\frac{x-T_i}{T_{i+k}-T_i}N_{i,k-1}(x)+\frac{T_{i+k+1}-x}{T_{i+k+1}-T_{i+1}}N_{i+1,k-1}(x).$"),
            (
                "prose",
                "The code repeats this recurrence for k=1,...,p. A term is set to zero "
                "when its denominator is zero, which is necessary at repeated knots. "
                "The returned matrix has one row per point and one column per basis "
                "function.",
            ),
            ("equation", r"$B(x)_{qi}=N_{i,p}(x),\qquad B(x)\in\mathbb{R}^{N\times n_{\mathrm{basis}}}.$"),
            ("heading", "Partition of unity"),
            ("equation", r"$\sum_i N_{i,p}(x)=1\quad\mathrm{for}\ x\in[0,1].$"),
            (
                "prose",
                "Compact support means only p+1 basis functions are nonzero at a generic "
                "point. The current implementation stores dense matrices, but the "
                "mathematical basis is local.",
            ),
        ],
    ),
    (
        "3. basis_and_second_derivative(x)",
        [
            (
                "prose",
                "This method returns the degree-p basis matrix B and its second spatial "
                "derivative Bdd. It evaluates bases of degrees p, p-1, and p-2, then "
                "uses exact B-spline derivative identities instead of differentiating "
                "the recursion with PyTorch autograd.",
            ),
            ("heading", "First derivative identity"),
            ("equation", r"$N'_{i,p}(x)=\frac{p}{T_{i+p}-T_i}N_{i,p-1}(x)-\frac{p}{T_{i+p+1}-T_{i+1}}N_{i+1,p-1}(x).$"),
            ("heading", "Second derivative identity"),
            ("equation", r"$N''_{i,p}(x)=a_iN'_{i,p-1}(x)-b_iN'_{i+1,p-1}(x),$"),
            ("equation", r"$a_i=\frac{p}{T_{i+p}-T_i},\qquad b_i=\frac{p}{T_{i+p+1}-T_{i+1}}.$"),
            (
                "prose",
                "The derivatives of the degree-(p-1) functions are expanded once more "
                "using the same first-derivative identity. Zero knot-span denominators "
                "again contribute zero. The method requires p >= 2 because the Poisson "
                "operator needs second derivatives.",
            ),
            ("equation", r"$B''(x)_{qi}=N''_{i,p}(x),\qquad B''(x)\in\mathbb{R}^{N\times n_{\mathrm{basis}}}.$"),
            (
                "note",
                "The basis and its derivatives depend on coordinates, knots, and degree, "
                "not on C. For fixed collocation points they can be computed once and "
                "reused during every optimizer epoch.",
            ),
        ],
    ),
    (
        "4. forward(xy): tensor-product field evaluation",
        [
            (
                "prose",
                "The input xy has shape (N,2). Its first and second columns are evaluated "
                "by the one-dimensional basis routine. Only interior basis columns and "
                "interior coefficients are retained.",
            ),
            ("equation", r"$B_x\in\mathbb{R}^{N\times(n_x-2)},\quad B_y\in\mathbb{R}^{N\times(n_y-2)},\quad C_I\in\mathbb{R}^{(n_x-2)\times(n_y-2)}.$"),
            ("heading", "Einstein summation used by the code"),
            ("equation", r"$u_q=\sum_{i=1}^{n_x-2}\sum_{j=1}^{n_y-2}(B_x)_{qi}(C_I)_{ij}(B_y)_{qj}.$"),
            (
                "code",
                'torch.einsum("ni,ij,nj->n", Bx, C_interior, By)',
            ),
            (
                "prose",
                "Equivalently, define a basis matrix at point q by Phi_q = bx_q outer "
                "by_q. Then u_q is the Frobenius inner product of Phi_q and C_I:",
            ),
            ("equation", r"$\Phi_q=(B_x)_{q,:}^{\mathrm{T}}(B_y)_{q,:},\qquad u_q=\langle\Phi_q,C_I\rangle_F.$"),
            ("heading", "Hard homogeneous boundary condition"),
            (
                "prose",
                "At x=0 only the omitted first x-basis is active; at x=1 only the omitted "
                "last x-basis is active. Thus u_h=0 on both vertical sides. The same "
                "argument applies at y=0 and y=1. No boundary penalty is needed.",
            ),
        ],
    ),
    (
        "5. f(x,y) and build_poisson_system(model, xy)",
        [
            ("heading", "f(x,y)"),
            (
                "prose",
                "This function evaluates the prescribed right-hand side at any tensors "
                "x and y. It is the Laplacian of the manufactured exact solution.",
            ),
            ("equation", r"$f(x,y)=-2\pi^2\sin(\pi x)\sin(\pi y).$"),
            ("heading", "Analytical Laplacian of the spline field"),
            ("equation", r"$\frac{\partial^2u_h}{\partial x^2}=\sum_{ij}C_{ij}N''_{i,p}(x)M_{j,p}(y),$"),
            ("equation", r"$\frac{\partial^2u_h}{\partial y^2}=\sum_{ij}C_{ij}N_{i,p}(x)M''_{j,p}(y).$"),
            ("equation", r"$\Delta u_h=\sum_{ij}C_{ij}\left[N''_{i,p}(x)M_{j,p}(y)+N_{i,p}(x)M''_{j,p}(y)\right].$"),
            (
                "prose",
                "At N collocation points, build_poisson_system forms the third-order "
                "tensor A_tensor[q,i,j] shown below, restricted to interior indices.",
            ),
            ("equation", r"$\mathcal{A}_{qij}=N''_{i,p}(x_q)M_{j,p}(y_q)+N_{i,p}(x_q)M''_{j,p}(y_q).$"),
            (
                "prose",
                "The last two axes are flattened, producing A with N rows and "
                "M=(n_x-2)(n_y-2) columns. The source vector is b_q=f(x_q,y_q).",
            ),
            ("equation", r"$A\in\mathbb{R}^{N\times M},\quad c=\mathrm{vec}(C_I)\in\mathbb{R}^{M},\quad r=Ac-b.$"),
        ],
    ),
    (
        "6. Configuration helper functions",
        [
            ("heading", "load_best_bspline_hyperparameters(path)"),
            (
                "prose",
                "This function opens a JSON file and reads n_basis and degree. It checks "
                "that the JSON root is an object, that both keys exist, and that their "
                "values can be converted to integers. It returns the pair "
                "(n_basis, degree).",
            ),
            (
                "code",
                '{\n'
                '  "n_basis": 20,\n'
                '  "degree": 3\n'
                '}',
            ),
            (
                "prose",
                "These are discrete model-architecture parameters. They define the "
                "dimension and smoothness of the approximation space; they are not "
                "optimized by gradient descent in this script.",
            ),
            ("heading", "select_device(device_name)"),
            (
                "prose",
                "Allowed inputs are cpu and cuda. CUDA is selected only if PyTorch reports "
                "an available CUDA device; otherwise the function raises an error instead "
                "of silently using the CPU. Every model tensor and collocation tensor is "
                "then created on the returned device.",
            ),
            ("equation", r"$\mathrm{device}\in\{\mathrm{CPU},\mathrm{CUDA}\}.$"),
            (
                "note",
                "CPU and GPU results need not be bit-for-bit identical. Floating-point "
                "reductions and some GPU kernels may use different operation orders, "
                "although the mathematical objective is unchanged.",
            ),
        ],
    ),
    (
        "7. main(): setup, collocation, and loss",
        [
            ("heading", "Model and collocation points"),
            (
                "prose",
                "main selects the device, loads tuned architecture values when requested, "
                "constructs the spline model, and samples n_interior random points "
                "uniformly in the unit square. The random seed makes a run reproducible "
                "on the same software and hardware configuration.",
            ),
            ("equation", r"$(x_q,y_q)\overset{\mathrm{iid}}{\sim}\mathcal{U}([0,1]^2),\qquad q=1,\ldots,N.$"),
            (
                "prose",
                "A and b are assembled once before training. This is the main computational "
                "saving over recomputing spline values and derivatives inside every epoch.",
            ),
            ("heading", "Nested poisson_residual_loss"),
            (
                "prose",
                "The nested loss function extracts and flattens the current interior "
                "coefficients, multiplies by the precomputed operator matrix, and returns "
                "the mean squared strong-form residual.",
            ),
            ("equation", r"$c=\mathrm{vec}(C[1:-1,1:-1]),\qquad r(c)=Ac-b,$"),
            ("equation", r"$\mathcal{L}_{\mathrm{PDE}}(c)=\frac{1}{N}\sum_{q=1}^{N}r_q(c)^2=\frac{1}{N}\|Ac-b\|_2^2.$"),
            (
                "prose",
                "Although the basis derivatives are analytical, autograd is still used "
                "to differentiate this scalar loss with respect to C. Because r is linear "
                "in c, its gradient is especially simple:",
            ),
            ("equation", r"$\nabla_c\mathcal{L}=\frac{2}{N}A^{\mathrm{T}}(Ac-b).$"),
        ],
    ),
    (
        "8. Optimization and stopping logic",
        [
            ("heading", "Adam optimizer"),
            (
                "prose",
                "The coefficient tensor is optimized with Adam at the chosen learning "
                "rate. Boundary entries have zero gradient because the loss only reads "
                "C[1:-1,1:-1]. The training helper repeatedly performs zero_grad, loss "
                "evaluation, backward differentiation, and optimizer.step.",
            ),
            ("heading", "ReduceLROnPlateau"),
            (
                "prose",
                "The scheduler monitors the training residual. If it does not improve for "
                "the configured patience, the learning rate is multiplied by 0.8, down "
                "to a minimum of 1e-5. Reducing the step size helps refine the coefficients "
                "after rapid initial progress.",
            ),
            ("equation", r"$\eta_{\mathrm{new}}=\max(0.8\eta,10^{-5}).$"),
            ("heading", "Early stopping"),
            (
                "prose",
                "The generic training routine tracks the best loss. An epoch counts as an "
                "improvement only if loss < best_loss - min_delta. Otherwise a patience "
                "counter advances; training stops after sufficiently many non-improving "
                "epochs. min_delta prevents insignificant numerical fluctuations from "
                "continually resetting the counter.",
            ),
            ("heading", "Why an optimizer is not mathematically necessary here"),
            (
                "prose",
                "For this linear Poisson equation and fixed basis, minimizing the residual "
                "is a linear least-squares problem. One could solve min ||Ac-b|| directly "
                "with QR or SVD. Adam is retained to share the generic training workflow "
                "with nonlinear PDE examples.",
            ),
            ("equation", r"$c^*=\mathrm{arg\,min}_c\|Ac-b\|_2^2.$"),
        ],
    ),
    (
        "9. Evaluation, errors, plots, and command line",
        [
            ("heading", "Evaluation grid"),
            (
                "prose",
                "After training, main constructs a regular n_grid by n_grid mesh, evaluates "
                "the learned spline through forward, and compares it with the exact field.",
            ),
            ("equation", r"$e_q=u_h(x_q,y_q)-u_{\mathrm{exact}}(x_q,y_q).$"),
            (
                "prose",
                "The script reports the maximum absolute error and a relative discrete "
                "L2 error:",
            ),
            ("equation", r"$\|e\|_{\infty}=\max_q|e_q|,$"),
            ("equation", r"$e_{L^2,\mathrm{rel}}=\frac{\left(\sum_q e_q^2\right)^{1/2}}{\left(\sum_q u_{\mathrm{exact},q}^2\right)^{1/2}}.$"),
            ("heading", "Figures"),
            (
                "prose",
                "Three surface plots show the predicted solution, exact solution, and "
                "absolute pointwise error. A fourth plot shows the loss history on a "
                "logarithmic vertical axis.",
            ),
            ("heading", "Command-line interface"),
            (
                "prose",
                "The __main__ block exposes device, architecture, JSON path, number of "
                "collocation points, epochs, learning rate, patience settings, reporting "
                "frequency, evaluation-grid size, and random seed. Arguments are passed "
                "to main without changing the mathematical formulation.",
            ),
            (
                "code",
                "python Poisson/Bspline_2D.py --device cpu\n"
                "python Poisson/Bspline_2D.py --device cuda --n-basis 20 --degree 3",
            ),
        ],
    ),
    (
        "10. Complete mathematical data flow",
        [
            ("heading", "Precomputation"),
            ("equation", r"$(x_q,y_q)\longrightarrow B_x,B_y,B''_x,B''_y,$"),
            ("equation", r"$\mathcal{A}_{qij}=B''_{x,qi}B_{y,qj}+B_{x,qi}B''_{y,qj},$"),
            ("equation", r"$A=\mathrm{reshape}(\mathcal{A}),\qquad b_q=f(x_q,y_q).$"),
            ("heading", "Training"),
            ("equation", r"$C_I\longrightarrow c=\mathrm{vec}(C_I)\longrightarrow r=Ac-b\longrightarrow \mathcal{L}=N^{-1}\|r\|_2^2.$"),
            ("heading", "Prediction"),
            ("equation", r"$u_h(x,y)=\sum_{i=1}^{n_x-2}\sum_{j=1}^{n_y-2}C_{ij}N_{i,p}(x)M_{j,p}(y).$"),
            (
                "prose",
                "The essential boundary condition is imposed by the approximation space, "
                "not by a penalty. Consequently the optimization objective contains only "
                "the interior PDE residual.",
            ),
            ("heading", "Shapes for N points and n_basis = n"),
            (
                "code",
                "xy                         : (N, 2)\n"
                "Bx, By, Bxx, Byy           : (N, n)\n"
                "interior basis matrices    : (N, n-2)\n"
                "C_interior                 : (n-2, n-2)\n"
                "operator tensor A_tensor   : (N, n-2, n-2)\n"
                "operator matrix A          : (N, (n-2)^2)\n"
                "coefficient vector c       : ((n-2)^2,)\n"
                "source and residual        : (N,)",
            ),
            (
                "note",
                "This separation between immutable basis data and trainable coefficients "
                "is the central design choice of the script.",
            ),
        ],
    ),
]


def main():
    """Generate the documentation PDF next to this build script."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    )
    with PdfPages(OUTPUT_PATH) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Mathematical documentation for Poisson/Bspline_2D.py"
        metadata["Author"] = "Generated from the implementation"
        metadata["Subject"] = "Tensor-product B-spline Poisson solver"
        for page_number, (title, blocks) in enumerate(PAGES, start=1):
            _add_page(pdf, title, blocks, page_number)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
