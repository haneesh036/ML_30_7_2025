import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


# Helper to parse Volume like '214.88K' or '1.67M' into numeric
def parse_volume(vol_str):
    """
    Convert volume strings with 'K' or 'M' suffix to integers.
    """
    if isinstance(vol_str, str):
        vol_str = vol_str.strip()
        if vol_str.endswith('K'):
            return float(vol_str[:-1].replace(',', '')) * 1e3
        if vol_str.endswith('M'):
            return float(vol_str[:-1].replace(',', '')) * 1e6
        # Fallback: remove commas
        return float(vol_str.replace(',', ''))
    # If already numeric
    return vol_str

# A1: Load IRCTC data and evaluate a kNN classifier
def load_irctc_data(filepath):
    """
    Load IRCTC stock price data from Excel, sorted by Date,
    parse Volume, then engineer a binary 'Direction' target:
      1 if next day's Price > today's Price, else 0.
    Returns feature matrix X and label array y.
    """
    df = pd.read_excel(filepath, sheet_name='IRCTC Stock Price')
    df.sort_values('Date', inplace=True)

    # Convert Volume column to numeric
    df['Volume'] = df['Volume'].apply(parse_volume)

    # Create target based on 'Price'
    df['NextClose'] = df['Price'].shift(-1)
    df.dropna(subset=['NextClose'], inplace=True)
    df['Direction'] = np.where(df['NextClose'] > df['Price'], 1, 0)

    # Select numeric features
    features = ['Open', 'High', 'Low', 'Price', 'Volume']
    X = df[features].values.astype(float)
    y = df['Direction'].astype(int).values
    return X, y

# A1 metric evaluation
def evaluate_classification_model(X, y, n_neighbors=5, test_size=0.3, random_state=42):
    rng = np.random.RandomState(random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    metrics = {}
    for split, true, pred in [('train', y_train, y_train_pred), ('test', y_test, y_test_pred)]:
        cm = confusion_matrix(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
        metrics[split] = {'confusion_matrix': cm, 'precision': precision, 'recall': recall, 'f1_score': f1}
    return metrics

#A2: Calculate MSE,RMSE,MAPE,R2 scores
def evaluate_price_prediction(actual, predicted):  # Function to calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2


# A3: Generate synthetic 2D data
def generate_synthetic_train(n_samples=20, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(1, 10, size=(n_samples, 2))
    y = np.where(X.sum(axis=1) > 11, 1, 0)
    return X, y

# A4/A5: Plot kNN decision boundaries on synthetic data
def classify_and_plot(X_train, y_train, k=3):
    xx = np.arange(0, 10.1, 0.1)
    yy = np.arange(0, 10.1, 0.1)
    xx_grid, yy_grid = np.meshgrid(xx, yy)
    grid_points = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    grid_pred = knn.predict(grid_points).reshape(xx_grid.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx_grid, yy_grid, grid_pred, alpha=0.3)
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='Class 0', edgecolor='k')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='Class 1', edgecolor='k')
    plt.title(f'kNN Classification (k={k})')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.legend()
    plt.show()

# A7: Hyperparameter tuning to find best k
def find_best_k(X, y, k_range=range(1, 21), cv=5):
    param_grid = {'n_neighbors': list(k_range)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='f1')
    grid.fit(X, y)
    return grid.best_params_['n_neighbors'], grid.best_score_

if __name__ == '__main__':
    filepath = 'Lab Session Data.xlsx'

    # A1: Load and evaluate
    X, y = load_irctc_data(filepath)
    metrics = evaluate_classification_model(X, y, n_neighbors=5)
    print('A1: Classification Metrics for k=5')
    for split in ['train', 'test']:
        print(f"{split.title()} Confusion Matrix:\n{metrics[split]['confusion_matrix']}")
        print(f"{split.title()} Precision: {metrics[split]['precision']:.3f}")
        print(f"{split.title()} Recall:    {metrics[split]['recall']:.3f}")
        print(f"{split.title()} F1-Score:  {metrics[split]['f1_score']:.3f}\n")

    #A2:
    df = pd.read_excel(filepath,sheet_name="Purchase data")

    # Features (A) and Target (C)
    A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    C = df['Payment (Rs)'].values

    # Split data into train and test sets
    A_train, A_test, C_train, C_test = train_test_split(A, C, test_size=0.2, random_state=42)

    # Compute pseudo-inverse only on training data
    L = np.linalg.pinv(A_train) @ C_train

    # Predictions
    predicted_C_train = A_train @ L
    predicted_C_test = A_test @ L

    # Evaluate metrics for training set
    train_mse, train_rmse, train_mape, train_r2 = evaluate_price_prediction(C_train, predicted_C_train)

    # Evaluate metrics for test set
    test_mse, test_rmse, test_mape, test_r2 = evaluate_price_prediction(C_test, predicted_C_test)

    # Print results
    print("\n--- Training Data ---")
    print("Mean Squared Error (MSE):", train_mse)
    print("Root Mean Squared Error (RMSE):", train_rmse)
    print("Mean Absolute Percentage Error (MAPE):", train_mape)
    print("R-squared (R²) Score:", train_r2)

    print("\n--- Test Data ---")
    print("Mean Squared Error (MSE):", test_mse)
    print("Root Mean Squared Error (RMSE):", test_rmse)
    print("Mean Absolute Percentage Error (MAPE):", test_mape)
    print("R-squared (R²) Score:", test_r2)

    # A3: Synthetic data
    X_syn, y_syn = generate_synthetic_train()
    plt.figure()
    plt.scatter(X_syn[y_syn==0,0], X_syn[y_syn==0,1], label='Class 0', edgecolor='k')
    plt.scatter(X_syn[y_syn==1,0], X_syn[y_syn==1,1], label='Class 1', edgecolor='k')
    plt.title('A3: Synthetic Training Data')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.legend() 
    plt.show()

    # A4/A5: Decision boundaries
    classify_and_plot(X_syn, y_syn, k=3)
    for k in [1, 6, 10]:
        classify_and_plot(X_syn, y_syn, k)

    # A7: Best k
    best_k, best_score = find_best_k(X, y)
    print(f'A7: Best k by GridSearchCV: {best_k} with F1-score = {best_score:.3f}')
