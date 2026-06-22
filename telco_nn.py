import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. FIXED TABULAR DATA PIPELINE (ELIMINATES NAN LEAKS)
# =====================================================================
def load_real_dataset(filepath='Telco-Churn.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find the dataset at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Secure numeric conversion and explicit non-inplace imputation
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Service data consolidation
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    df_service_numeric = df[service_cols].replace({'Yes': 1, 'No': 0})
    df['num_services'] = df_service_numeric.sum(axis=1).astype(float)
    
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                   labels=['0-1yrs', '1-2yrs', '2-4yrs', '4-6yrs'],
                                   right=False, include_lowest=True)

    X = df.drop(columns=['customerID', 'Churn']).copy()
    y = df['Churn'].values

    # Explicit column tracking
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'num_services']
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
    
    # Cast categoricals safely to clean strings
    for c in cat_cols:
        X[c] = X[c].astype(str)

    # Stratified Splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    # Transform numerical and categorical elements
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='drop')

    X_tr = pre.fit_transform(X_train).T
    X_vl = pre.transform(X_val).T
    X_te = pre.transform(X_test).T
    
    # Final check to guarantee absolutely zero NaNs remain in arrays
    X_tr = np.nan_to_num(X_tr)
    X_vl = np.nan_to_num(X_vl)
    X_te = np.nan_to_num(X_te)
    
    Y_tr = y_train.reshape(1, -1)
    Y_vl = y_val.reshape(1, -1)
    Y_te = y_test.reshape(1, -1)

    pos_weight = float((Y_tr == 0).sum() / (Y_tr == 1).sum())
    
    print("--- Real Telco Dataset Preprocessing Complete ---")
    print(f"Features: {X_tr.shape[0]} | Train Rows: {X_tr.shape[1]} | Val Rows: {X_vl.shape[1]} | Test Rows: {X_te.shape[1]}")
    print(f"Calculated scale_pos_weight: {pos_weight:.4f}\n")
    
    return X_tr, X_vl, X_te, Y_tr, Y_vl, Y_te, pos_weight

# =====================================================================
# 2. ACTIVATIONS & STABLE INITIALIZATION
# =====================================================================
def relu(z):          
    return np.maximum(0.0, z)

def relu_back(dA, z): 
    d = dA.copy()
    d[z <= 0] = 0.0
    return d

def sigmoid(z):       
    return 1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0)))

def init_params(dims):
    np.random.seed(42)
    p = {}
    L = len(dims) - 1
    for l in range(1, L + 1):
        p[f'W{l}'] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2.0 / dims[l-1])
        p[f'b{l}'] = np.zeros((dims[l], 1))
    return p

# =====================================================================
# 3. PROPAGATION ENGINE (WITH REINFORCED CLIPPING)
# =====================================================================
def forward(X, params):
    L = sum(1 for k in params if k.startswith('W'))
    cache = {'A0': X}

    for l in range(1, L):
        cache[f'Z{l}'] = params[f'W{l}'] @ cache[f'A{l-1}'] + params[f'b{l}']
        cache[f'A{l}'] = relu(cache[f'Z{l}'])

    cache[f'Z{L}'] = params[f'W{L}'] @ cache[f'A{L-1}'] + params[f'b{L}']
    cache[f'A{L}'] = sigmoid(cache[f'Z{L}'])
    return cache[f'A{L}'], cache, L

def backward(params, cache, Y, L, pos_weight=1.0):
    grads = {}
    m = Y.shape[1]
    
    A_out = np.clip(cache[f'A{L}'], 1e-7, 1.0 - 1e-7)

    dZ = (A_out - Y) * ((1.0 - Y) + Y * pos_weight) / m
    dZ = np.nan_to_num(np.clip(dZ, -2.0, 2.0))
    
    grads[f'dW{L}'] = np.clip(dZ @ cache[f'A{L-1}'].T, -2.0, 2.0)
    grads[f'db{L}'] = np.clip(dZ.sum(axis=1, keepdims=True), -2.0, 2.0)
    dA = params[f'W{L}'].T @ dZ

    for l in reversed(range(1, L)):
        dZ = relu_back(dA, cache[f'Z{l}'])
        dZ = np.nan_to_num(np.clip(dZ, -2.0, 2.0))
        
        grads[f'dW{l}'] = np.clip(dZ @ cache[f'A{l-1}'].T, -2.0, 2.0)
        grads[f'db{l}'] = np.clip(dZ.sum(axis=1, keepdims=True), -2.0, 2.0)
        dA = params[f'W{l}'].T @ dZ

    return grads

def stable_bce_loss(Z_out, Y, pos_weight=1.0):
    Z_safe = np.nan_to_num(np.clip(Z_out, -12.0, 12.0))
    w = np.where(Y == 1, pos_weight, 1.0)
    loss = w * (np.maximum(0, Z_safe) - Z_safe * Y + np.log(1.0 + np.exp(-np.abs(Z_safe))))
    return np.mean(loss)

# =====================================================================
# 4. STEP OPTIMIZERS
# =====================================================================
def sgd_init(params): return {}
def sgd_step(params, grads, state, t, lr):
    for k in params:
        if 'd'+k in grads: 
            params[k] -= lr * np.nan_to_num(grads['d'+k])
    return params, state

def momentum_init(params, beta=0.9):
    return {'beta': beta, **{k: np.zeros_like(v) for k, v in params.items()}}
def momentum_step(params, grads, state, t, lr):
    b = state['beta']
    for k in params:
        if 'd'+k in grads:
            g = np.nan_to_num(grads['d'+k])
            state[k] = b * state[k] + (1.0 - b) * g
            params[k] -= lr * state[k]
    return params, state

def adam_init(params, beta1=0.9, beta2=0.999):
    s = {'beta1': beta1, 'beta2': beta2}
    for k in params:
        s['v_'+k] = np.zeros_like(params[k])
        s['s_'+k] = np.zeros_like(params[k])
    return s
def adam_step(params, grads, state, t, lr, eps=1e-8):
    b1, b2 = state['beta1'], state['beta2']
    for k in params:
        if 'd'+k in grads:
            g = np.nan_to_num(grads['d'+k])
            state['v_'+k] = b1 * state['v_'+k] + (1.0 - b1) * g
            state['s_'+k] = b2 * state['s_'+k] + (1.0 - b2) * g**2
            vc = state['v_'+k] / (1.0 - b1**t)
            sc = state['s_'+k] / (1.0 - b2**t)
            params[k] -= lr * vc / (np.sqrt(sc) + eps)
    return params, state

# =====================================================================
# 5. METRICS ENGINE
# =====================================================================
def roc_auc(y_true, y_prob):
    order = np.argsort(-y_prob)
    yt = y_true[order]
    n_pos = yt.sum()
    n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0: 
        return 0.5
    tp = fp = auc = 0
    for label in yt:
        if label == 1: tp += 1
        else:          fp += 1; auc += tp
    return auc / (n_pos * n_neg + 1e-8)

# =====================================================================
# 6. RUNNER ENGINE
# =====================================================================
def train(X_tr, Y_tr, X_val, Y_val, dims, opt_init, opt_step, lr, epochs, batch_size, pos_weight, name):
    params = init_params(dims)
    state  = opt_init(params)
    m      = X_tr.shape[1]
    t      = 0
    hist   = {'loss': [], 'val_auc': []}

    for ep in range(1, epochs + 1):
        idx = np.random.permutation(m)
        Xs, Ys = X_tr[:, idx], Y_tr[:, idx]

        for i in range(0, m, batch_size):
            Xb, Yb = Xs[:, i:i+batch_size], Ys[:, i:i+batch_size]
            _, cache, L = forward(Xb, params)
            grads       = backward(params, cache, Yb, L, pos_weight)
                
            t += 1
            params, state = opt_step(params, grads, state, t, lr)

        _, cache_tr, L = forward(X_tr, params)
        A_val, _, _    = forward(X_val, params)
        
        loss_val = stable_bce_loss(cache_tr[f'Z{L}'], Y_tr, pos_weight)
        auc_val  = roc_auc(Y_val.flatten(), A_val.flatten())
        
        hist['loss'].append(loss_val)
        hist['val_auc'].append(auc_val)

        if ep % 10 == 0 or ep == 1:
            print(f"  [{name:<15}] Epoch {ep:2d} | Stable Loss: {loss_val:.4f} | Val ROC-AUC: {auc_val:.4f}")

    return params, hist

if __name__ == '__main__':
    # 1. Dynamically target local environment directory path
    DATASET_PATH = 'C:/Users/HP/Desktop/Telco-churn-prediction/data/Telco-Churn.csv'
    if not os.path.exists(DATASET_PATH):
        DATASET_PATH = 'Telco-Churn.csv'
        
    X_tr, X_val, X_te, Y_tr, Y_val, Y_te, pos_w = load_real_dataset(DATASET_PATH)
    
    DIMS = [X_tr.shape[0], 16, 8, 1]
    EPOCHS = 80
    
    strategies = [
        ('Batch GD',      sgd_init,      sgd_step,      0.01,   X_tr.shape[1]),
        ('Mini-Batch GD', sgd_init,      sgd_step,      0.005,  64),
        ('SGD + Momentum',momentum_init, momentum_step, 0.005,  64),
        ('Adam',          adam_init,     adam_step,     0.0005, 64)
    ]
    
    results = {}
    trained_models = {}
    
    for name, init_fn, step_fn, lr, bs in strategies:
        print(f"Beginning Experiment Sweep: {name}...")
        final_params, history = train(X_tr, Y_tr, X_val, Y_val, DIMS, init_fn, step_fn, 
                                      lr, EPOCHS, bs, pos_weight=pos_w, name=name)
        results[name] = history
        trained_models[name] = final_params
        print()
        
    print("=" * 70)
    print("                 FINAL PERFORMANCE METRICS (REAL TEST SET)     ")
    print("=" * 70)
    for name in trained_models:
        A_test, _, _ = forward(X_te, trained_models[name])
        test_auc_score = roc_auc(Y_te.flatten(), A_test.flatten())
        print(f" >> Optimization Strategy: {name:<15} | Final Test Set ROC-AUC Score: {test_auc_score:.4f}")
    print("=" * 70)
    
    # Export curves to track progression path
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optimization Gradient Sweep Analysis Matrix', fontsize=12, fontweight='bold')
    for name in results:
        ax1.plot(results[name]['loss'], label=name, lw=2)
        ax2.plot(results[name]['val_auc'], label=name, lw=2)
    ax1.set_title('BCE Loss Convergence Path'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title('Validation ROC-AUC Trend Line'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('AUC Score')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('telco_optimization_comparison.png', dpi=150)
    print("\nPlot exported successfully -> 'telco_optimization_comparison.png'")
