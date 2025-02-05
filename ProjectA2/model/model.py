import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px

class MyRegression:
    def __init__(self, regularization: float = None, lr: float = 0.01, method: str = 'mini_batch',
                 weight_init: str = 'zeros', batch_size: int = 64, n_epochs: int = 100, momentum: float = None,
                 poly_degree: int = None, cv: int = 5, log_transform: bool = False, use_mlflow: bool = False):
        self.lr = lr  # learning rate
        self.method = method  # optimization method
        self.weight_init = weight_init  # weight initialization type
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.regularization = regularization  # Ridge or Lasso
        self.momentum = momentum  # momentum coefficient
        self.use_mlflow = use_mlflow
        self.poly_degree = poly_degree
        self.cv = cv  # K-fold validation
        self.weight_decay = 1e-5  # L2 regularization
        self.log_transform = log_transform

        valid_method = ['mini_batch', 'stochastic', 'batch']
        valid_weight = ['normal', 'xavier', 'zeros']

        if self.method not in valid_method:
            raise ValueError(f'method must be in {valid_method}')
        if self.weight_init not in valid_weight:
            raise ValueError(f'weight_init must be in {valid_weight}')

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.astype('float').values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.astype(float).values

        X = np.array(X).astype('float')
        y = np.array(y).astype('float')

        if self.log_transform:
            y = np.log1p(y)

        if self.poly_degree is not None:
            X = self._polynomial_features(X)

        intercept = np.ones((X.shape[0], 1)).astype('float')
        X = np.concatenate((intercept, X), axis=1)

        patience = 4
        epoch_without_improvement = 0
        self.kfold_train_accuracy_list = []
        self.kfold_train_loss_list = []

        cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            self.W = self.weight_initializer(X.shape[1])

            x_train_cross = X[train_idx]
            y_train_cross = y[train_idx]
            x_val_cross = X[val_idx]
            y_val_cross = y[val_idx]

            self.old_loss = np.inf

            for epoch in range(self.n_epochs):
                perm_idx = np.random.permutation(len(x_train_cross))
                x_train_cross = x_train_cross[perm_idx]
                y_train_cross = y_train_cross[perm_idx]

                if self.method == 'mini_batch':
                    for i in range(0, len(x_train_cross), self.batch_size):
                        batch_X = x_train_cross[i:i + self.batch_size]
                        batch_y = y_train_cross[i:i + self.batch_size]
                        train_loss, train_accuracy = self._train(batch_X, batch_y)
                elif self.method == 'stochastic':
                    for i in range(x_train_cross.shape[0]):
                        batch_X = x_train_cross[i:i + 1]
                        batch_y = y_train_cross[i:i + 1]
                        train_loss, train_accuracy = self._train(batch_X, batch_y)
                else:
                    train_loss, train_accuracy = self._train(x_train_cross, y_train_cross)

                self.kfold_train_accuracy_list.append(train_accuracy)
                self.kfold_train_loss_list.append(train_loss)

                y_hat_val = self._predict(x_val_cross, is_train=True)
                val_loss, val_accuracy = self.mse(y_val_cross, y_hat_val), self.r2_score(y_val_cross, y_hat_val)

                if np.isclose(self.old_loss, val_loss, atol=1e-3):
                    epoch_without_improvement += 1
                    if epoch_without_improvement >= patience:
                        print('early stopping')
                        break
                else:
                    epoch_without_improvement = 0

                self.old_loss = val_loss

            print(
                f"{fold} --> train_loss: {np.mean(self.kfold_train_loss_list):.3f} | train_accuracy: {np.mean(self.kfold_train_accuracy_list):.4f} | val_loss: {val_loss:.4f} | val_accuracy: {val_accuracy:.4f}")

    def _predict(self, X, is_train=False):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.astype('float').values
        elif isinstance(X, np.ndarray):
            X = np.array(X).astype('float')

        if not is_train and self.poly_degree is not None:
            X = self._polynomial_features(X)

        if X.shape[1] == self.W.shape[0] - 1:  # Check if bias is missing
            intercept = np.ones((X.shape[0], 1)).astype('float')
            X = np.concatenate((intercept, X), axis=1)

        return np.dot(X, self.W.reshape(-1, 1)).flatten()

    def _train(self, X, y):
        y_hat = self._predict(X, is_train=True)

        m = X.shape[0]

        grad = (1 / m) * np.dot(X.T, (y_hat - y).reshape(-1, 1))

        grad += self.weight_decay * self.W.reshape(-1, 1)

        grad = np.clip(grad, -5.0, 5.0)

        if self.momentum is not None:
            if not hasattr(self, 'velocity'):
                self.velocity = np.zeros_like(self.W)
            self.velocity = self.momentum * self.velocity - self.lr * grad.flatten()  # Ensure 1D shape
            self.W += self.velocity.flatten()
        else:
            self.W -= self.lr * grad.flatten()

        if np.isnan(self.W).any() or np.isinf(self.W).any():
            print("Warning: NaN or Inf detected in weights!")
            self.W = np.where(np.isnan(self.W) | np.isinf(self.W), np.random.randn(*self.W.shape) * 0.01, self.W)

        return self.mse(y, y_hat), self.r2_score(y, y_hat)

    def mse(self, ytrue, ypred):
        return np.sum((ypred - ytrue) ** 2) / ytrue.shape[0]

    def r2_score(self, ytrue, ypred):
        rss = np.sum((ytrue - ypred) ** 2)
        tss = np.sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - (rss / tss)

    def weight_initializer(self, num_features):
        if self.weight_init == 'zeros':
            return np.zeros(num_features).astype('float')
        elif self.weight_init == 'xavier':
            limit = 1 / np.sqrt(num_features)
            return np.random.uniform(-limit, limit, size=num_features).astype('float')
        else:
            return np.random.randn(num_features).astype('float') * 0.01

    def _polynomial_features(self, X):

        if self.poly_degree is None:
            return X

        X = np.array(X, dtype='float')  # Ensure NumPy format
        n_samples, n_features = X.shape
        poly_features = [X]

        # Add higher-degree features (x^2, x^3, etc.)
        for degree in range(2, self.poly_degree + 1):
            poly_features.append(X ** degree)

        # Add interaction terms for degree = 2
        # if self.poly_degree == 2:
        #     interaction_terms = [np.prod(X[:, np.array(combo)], axis=1) for combo in combinations(range(n_features), 2)]
        #     return np.column_stack(poly_features + interaction_terms)

        return np.concatenate(poly_features, axis=1)

    def _coef(self):
        """Returns all weights except the intercept (bias)."""
        return self.W[1:]

    def _bias(self):
        """Returns the bias term."""
        return self.W[0]

    def plot_feature_importance(self, feature_names=None):

        # Ensure weights are NumPy arrays
        feature_importance = np.array(self._coef())

        # If feature names are not provided, generate default names
        num_features = len(feature_importance)
        if feature_names is None:
            feature_names = [f'Feature {i + 1}' for i in range(num_features)]

        # Adjust feature names for polynomial features (No interaction terms now)
        expanded_feature_names = self.get_expanded_feature_names(feature_names)

        # Dictionary to sum importance across polynomial degrees
        feature_weight_dict = {name: 0 for name in expanded_feature_names}

        # Aggregate absolute importance values for polynomial terms
        for name, weight in zip(expanded_feature_names, feature_importance):
            feature_weight_dict[name] += abs(weight)  # Sum absolute importance

        # Convert to DataFrame and sort by importance
        sorted_features = sorted(feature_weight_dict.items(), key=lambda x: x[1], reverse=True)
        coefs_df = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])

        # Normalize feature importance for visualization
        max_importance = coefs_df["Importance"].max()
        min_importance = coefs_df["Importance"].min()

        if max_importance > min_importance:
            normalized_importance = (coefs_df["Importance"] - min_importance) / (max_importance - min_importance)
            color_indices = (normalized_importance * 255).astype(int)  # Convert to color scale (0-255)
        else:
            color_indices = np.full(len(coefs_df), 128)  # Default middle shade if importance is constant

        colors = [plt.cm.Greens(idx / 255) for idx in color_indices]

        # Plot feature importance
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=coefs_df["Importance"], y=coefs_df["Feature"], palette=colors)

        max_importance = np.max(np.abs(coefs_df["Importance"]))

        # Add data labels with proper positioning
        for index, value in enumerate(coefs_df["Importance"]):
            label = f"{value:.4f}"
            if abs(value) < 0.05 * max_importance:  # If bar is **too short**
                ax.text(value + (0.05 * max_importance), index, label,
                        va='center', ha='left', fontsize=12, color='black', fontweight='bold')
            else:
                ax.text(value * 0.95, index, label,
                        va='center', ha='right' if value > 0 else 'left', fontsize=9, color='white', fontweight='bold')

                # Add labels and title
        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title('Feature Importance Plot', fontsize=16)
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        # Show the plot
        plt.show()

    def get_expanded_feature_names(self, feature_names):

        if self.poly_degree is None or self.poly_degree == 1:
            return feature_names  # No transformation, return as is

        expanded_feature_names = []

        # Generate polynomial feature names (x, x^2, x^3, ...)
        for degree in range(1, self.poly_degree + 1):
            expanded_feature_names.extend([f"{col}^{degree}" for col in feature_names])

        return expanded_feature_names


# %%
class L1Penalty:
    """
    Implements L1 (Lasso) regularization.
    """

    def __init__(self, l):
        self.l = l

    def __call__(self, W):
        return self.l * np.sum(np.abs(W))

    def derivation(self, W):
        return self.l * np.sign(W)


class L2Penalty:
    """
    Implements L2 (Ridge) regularization.
    """
    def __init__(self, l):
        self.l = l

    def __call__(self, W):
        return self.l * np.sum(np.square(W))

    def derivation(self, W):
        return self.l * 2 * W


class LassoRegression(MyRegression):
    """
    Implements Lasso Regression using L1 Regularization.
    """

    def __init__(self, l=0.1, **kwargs):
        super().__init__(regularization=L1Penalty(l=l), **kwargs)


class RidgeRegression(MyRegression):
    """
    Implements Ridge Regression using L2 Regularization.
    """
    def __init__(self, l=0.1, **kwargs):
        super().__init__(regularization=L2Penalty(l=l), **kwargs)


def get_expanded_feature_names(feature_names, poly_degree):
    """
    Expands feature names based on polynomial degrees (x, x^2, x^3, ...).

    Args:
    - feature_names (list): Original feature names.
    - poly_degree (int): Polynomial degree used in the model.

    Returns:
    - expanded_feature_names (list): Updated feature names with polynomial degrees.
    """
    if poly_degree is None or poly_degree == 1:
        return feature_names  # No transformation, return as is

    expanded_feature_names = []

    # Generate polynomial feature names (x, x^2, x^3, ...)
    for degree in range(1, poly_degree + 1):
        expanded_feature_names.extend([f"{col}^{degree}" for col in feature_names])

    return expanded_feature_names


def plot_feature_importance(coefficients, feature_names, poly_degree=None):
    """
    Plots feature importance using Plotly horizontal bar chart, considering polynomial features.

    Parameters:
    - coefficients (np.array): Model coefficients
    - feature_names (list): Corresponding feature names
    - poly_degree (int): Polynomial degree used in model

    Returns:
    - fig: Plotly figure
    """

    # Ensure coefficients are NumPy arrays
    feature_importance = np.array(coefficients)

    # Expand feature names to account for polynomial features
    expanded_feature_names = get_expanded_feature_names(feature_names, poly_degree)

    # Dictionary to sum importance across polynomial degrees
    feature_weight_dict = {name.split("^")[0]: 0 for name in expanded_feature_names}

    # Aggregate absolute importance values across different polynomial terms
    for name, weight in zip(expanded_feature_names, feature_importance):
        base_feature = name.split("^")[0]  # Extract base feature name
        feature_weight_dict[base_feature] += abs(weight)

    # Convert to DataFrame and sort by importance
    coefs_df = pd.DataFrame(sorted(feature_weight_dict.items(), key=lambda x: x[1], reverse=True),
                            columns=["Feature", "Importance"])

    # Define Green Shades for Visualization
    green_shades = [
        "#0B3D02", "#0D5F00", "#145A32", "#166D25", "#1B8A3B", "#1D8348", "#229954", "#27AE60", "#2ECC71",
        "#34D058", "#39D27F", "#3AE87D", "#42F59E", "#58D68D", "#5BE395", "#70EF9C", "#82E0AA", "#A9DFBF",
        "#C8F7C5", "#D5F5E3"
    ]

    # Normalize importance for color scaling
    max_importance = coefs_df["Importance"].max()
    min_importance = coefs_df["Importance"].min()

    if max_importance > min_importance:
        normalized_importance = (coefs_df["Importance"] - min_importance) / (max_importance - min_importance)
        color_indices = (normalized_importance * (len(green_shades) - 1)).astype(int)
    else:
        color_indices = np.full(len(coefs_df), len(green_shades) // 2)

    colors = [green_shades[idx] for idx in color_indices]

    # Create Plotly Horizontal Bar Chart
    fig = px.bar(
        coefs_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=coefs_df["Importance"].round(4),
        color=coefs_df["Importance"],
        color_continuous_scale=green_shades,
        labels={"Importance": "Feature Importance", "Feature": "Features"},
        title="Feature Importance Plot"
    )

    fig.update_traces(marker=dict(line=dict(color="black", width=1)))  # Border for bars
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Importance",
        yaxis_title="Features",
        coloraxis_showscale=False  # Hide color bar
    )

    return fig