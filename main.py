# ================================================================
# ELL784 / ELL7286  ─  Assignment 3
# "Fast Time-as-Channel LSGAN + DANN for Soli Gesture Recognition"
# Google Colab Notebook  |  IIT Delhi
# ================================================================
# SETUP:
#   1. Runtime > Change runtime type > T4 GPU
#   2. Run !pip install -q gdown seaborn h5py
#   3. Set DATA_DIR to your extracted dataset path
#   4. Run All
# ================================================================

# ── Cell 1 | Install ────────────────────────────────────────────
# !pip install -q seaborn h5py

# ── Cell 2 | Imports & Config ───────────────────────────────────
import os, random, warnings
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import (Dataset, DataLoader,
                               ConcatDataset, TensorDataset, Subset)
from sklearn.metrics import (f1_score, classification_report,
                              confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'figure.facecolor': 'white'})

# ── Reproducibility ─────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataset path ─────────────────────────────────────────────────
DATA_DIR = '/content/extracted_files/dsp'   # ← change if your H5 files are elsewhere

# ── Hyper-parameters ─────────────────────────────────────────────
CFG = dict(
    num_classes     = 11,    # FIXED: Dropped redundant 12th class
    num_subjects    = 16,    # dynamic splitting will handle any number of subjects
    T               = 40,    # temporal frames per sample (subsampled for speed)
    R               = 32,    # range bins
    D               = 32,    # doppler bins  (1024 = 32×32)

    # Fine-grained class indices
    fine_grained    = [0, 1, 2, 3],

    # Training
    batch_size      = 32,
    epochs_gan      = 50,    # Fast LSGAN converges quickly
    epochs_main     = 80,    # INCREASED to 80 for better fine-grained separation
    lr_G            = 1e-4,
    lr_D            = 1e-4,
    lr_main         = 3e-4,
    betas           = (0.5, 0.9),

    # LSGAN
    n_critic        = 1,     # 1:1 training ratio for speed

    # DANN
    lambda_domain   = 0.3,

    # Augmentation
    latent_dim      = 128,
    n_aug           = 300,   # synthetic sequences per fine-grained class

    # Feature extractor
    feat_dim        = 256,
)

# FIXED: Removed 'Air CCW'
GESTURE_NAMES = [
    'Finger Slider', 'Finger Rub', 'Pinch Index', 'Pinch Pinky',
    'Tap/Click', 'Full Pinch', 'Push Wave', 'Pull Wave',
    'Palm Hold', 'Side Tap', 'Air CW'
]

# ── Cell 3 | Data Loading ────────────────────────────────────────
def load_h5_dataset(data_dir: str):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    print(f"Loading {len(files)} files ...")

    Xs, ys, ss = [], [], []
    for fname in tqdm(files, desc='Reading H5'):
        parts   = fname.replace('.h5', '').split('_')
        gesture = int(parts[0])
        subject = int(parts[1])

        # FILTER: Drop the redundant class (Class 11: Air CCW)
        if gesture == 11:
            continue

        fpath = os.path.join(data_dir, fname)
        with h5py.File(fpath, 'r') as f:
            rd = np.stack([f['ch0'][:], f['ch1'][:],
                           f['ch2'][:], f['ch3'][:]]).mean(axis=0)
            T_actual = rd.shape[0]
            rd = rd.reshape(T_actual, 32, 32).astype(np.float32)

        T_target = CFG['T']
        if T_actual >= T_target:
            rd = rd[:T_target]
        else:
            pad = np.zeros((T_target - T_actual, 32, 32), dtype=np.float32)
            rd = np.concatenate([rd, pad], axis=0)

        Xs.append(rd)
        ys.append(gesture)
        ss.append(subject)

    X = np.stack(Xs)
    y = np.array(ys, dtype=np.int64)
    s = np.array(ss, dtype=np.int64)
    print(f"Loaded | X: {X.shape}  classes: {np.unique(y)}  subjects: {np.unique(s)}")
    return X, y, s

def normalize_rd(X: np.ndarray) -> np.ndarray:
    X = np.log1p(np.abs(X))
    mu  = X.mean(axis=(1, 2, 3), keepdims=True)
    std = X.std (axis=(1, 2, 3), keepdims=True) + 1e-6
    return (X - mu) / std

def load_dataset(data_dir: str):
    X, y, s = load_h5_dataset(data_dir)
    X = normalize_rd(X)
    return X, y, s

# ── Cell 4 | Dataset Class ───────────────────────────────────────
class SoliDataset(Dataset):
    def __init__(self, X, y, subjects):
        self.X        = torch.from_numpy(X.astype(np.float32))
        self.y        = torch.from_numpy(y.astype(np.int64))
        self.subjects = torch.from_numpy(subjects.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)    # (1, T, R, D)
        return x, self.y[idx], self.subjects[idx]

class AugWrapper(Dataset):
    def __init__(self, td):
        self.td = td
    def __len__(self):
        return len(self.td)
    def __getitem__(self, i):
        return self.td[i]

# ── Cell 5 | Model Architecture ─────────────────────────────────
class FeatureExtractor(nn.Module):
    """
    Extracts spatial features while preserving the temporal dimension.
    Outputs sequence of features: (B, T_prime, feat_dim)
    """
    def __init__(self, feat_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16), nn.LeakyReLU(0.2, True),
            nn.MaxPool3d((2, 2, 2)), # T=40 -> 20

            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.LeakyReLU(0.2, True),
            nn.MaxPool3d((2, 2, 2)), # T=20 -> 10

            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64), nn.LeakyReLU(0.2, True),
            nn.MaxPool3d((2, 2, 2)), # T=10 -> 5

            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.LeakyReLU(0.2, True),
            
            # Pool spatial dimensions (R, D) to 1x1, but leave T alone.
            nn.AdaptiveAvgPool3d((None, 1, 1)), 
        )
        self.proj = nn.Sequential(
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        features = self.cnn(x)                                      # (B, 128, T', 1, 1)
        features = features.squeeze(-1).squeeze(-1).transpose(1, 2) # (B, T', 128)
        
        B, T_prime, C = features.shape
        features = self.proj(features.reshape(B * T_prime, C))
        return features.view(B, T_prime, -1)                        # (B, T', feat_dim)

class GestureClassifier(nn.Module):
    """Processes the temporal feature sequence with an LSTM for final classification."""
    def __init__(self, feat_dim=256, num_classes=11):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, 128, batch_first=True)
        self.net = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, feat):
        out, (hn, cn) = self.lstm(feat)
        final_out = out[:, -1, :] 
        return self.net(final_out)

class _GRL(autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        lam = ctx.saved_tensors[0].item()
        return -lam * grad, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return _GRL.apply(x, self.lam)

class DomainClassifier(nn.Module):
    """Processes the temporal sequence to predict subject ID (DANN)."""
    def __init__(self, feat_dim=256, num_subjects=16):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.lstm = nn.LSTM(feat_dim, 128, batch_first=True)
        self.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_subjects),
        )

    def forward(self, feat, lam=1.0):
        self.grl.lam = lam
        feat = self.grl(feat)
        out, _ = self.lstm(feat)
        final_out = out[:, -1, :]
        return self.net(final_out)

class FastSequenceGenerator(nn.Module):
    """Generates T frames simultaneously by treating Time as the Channel dimension."""
    def __init__(self, latent_dim=128, num_classes=11, T=40, R=32, D=32):
        super().__init__()
        self.T, self.R, self.D = T, R, D
        self.label_emb = nn.Embedding(num_classes, 16)
        
        in_dim = latent_dim + 16
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8), nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            
            nn.Conv2d(64, T, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x   = torch.cat([z, emb], dim=1)
        x   = self.fc(x).view(-1, 256, 8, 8)
        out = self.deconv(x)                  # (B, 40, 32, 32)
        return out.unsqueeze(1)               # (B, 1, 40, 32, 32)

class FastSequenceDiscriminator(nn.Module):
    """Evaluates realism by looking at all 40 frames as input channels."""
    def __init__(self, num_classes=11, T=40, R=32, D=32):
        super().__init__()
        self.T, self.R, self.D = T, R, D
        self.label_emb = nn.Embedding(num_classes, R * D)
        
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(T + 1, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(256 * 4 * 4, 1)),
        )

    def forward(self, x, labels):
        x = x.squeeze(1) # (B, 40, 32, 32)
        lmap = self.label_emb(labels).view(-1, 1, self.R, self.D)
        inp  = torch.cat([x, lmap], dim=1) # (B, 41, 32, 32)
        return self.net(inp)

# ── Cell 6 | Loss & Schedule ─────────────────────────────────────
def dann_schedule(epoch, total, gamma=10.0):
    p = epoch / total
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0

# ── Cell 7 | Fast LSGAN Training ─────────────────────────────────
def train_lsgan(G, D_gan, dataloader, cfg, device):
    opt_G = torch.optim.Adam(G.parameters(),     lr=cfg['lr_G'], betas=cfg['betas'])
    opt_D = torch.optim.Adam(D_gan.parameters(), lr=cfg['lr_D'], betas=cfg['betas'])
    mse   = nn.MSELoss() 

    G.train(); D_gan.train()
    g_losses, d_losses = [], []

    for epoch in range(1, cfg['epochs_gan'] + 1):
        eg, ed, nb = 0., 0., 0

        for batch_x, batch_y, _ in dataloader:
            real_frames = batch_x.to(device)   
            labels      = batch_y.to(device)
            B           = real_frames.size(0)

            # 1. Train Discriminator
            z     = torch.randn(B, cfg['latent_dim'], device=device)
            fake  = G(z, labels)
            
            d_real = D_gan(real_frames, labels)
            d_fake = D_gan(fake.detach(), labels)
            loss_D = 0.5 * mse(d_real, torch.ones_like(d_real)) + \
                     0.5 * mse(d_fake, torch.zeros_like(d_fake))
                     
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # 2. Train Generator
            d_fake_g = D_gan(fake, labels)
            loss_G   = 0.5 * mse(d_fake_g, torch.ones_like(d_fake_g))
            
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            eg += loss_G.item(); ed += loss_D.item(); nb += 1

        g_losses.append(eg / nb); d_losses.append(ed / nb)
        if epoch % 10 == 0:
            print(f"  GAN epoch {epoch:3d}/{cfg['epochs_gan']} | "
                  f"D: {d_losses[-1]:.4f}  G: {g_losses[-1]:.4f}")

    return G, g_losses, d_losses

# ── Cell 8 | Augment Fine-grained Classes ───────────────────────
@torch.no_grad()
def generate_augmented_sequences(G, fine_grained_classes, n_aug, cfg, device):
    G.eval()
    aug_X, aug_y, aug_s = [], [], []

    for cls in fine_grained_classes:
        labels = torch.full((n_aug,), cls, dtype=torch.long, device=device)
        z      = torch.randn(n_aug, cfg['latent_dim'], device=device)
        seqs   = G(z, labels).cpu()                          # (n_aug, 1, T, R, D) natively
        
        aug_X.append(seqs)
        aug_y.append(torch.full((n_aug,), cls, dtype=torch.long))
        aug_s.append(torch.full((n_aug,), -1,  dtype=torch.long))

    aug_X = torch.cat(aug_X)
    aug_y = torch.cat(aug_y)
    aug_s = torch.cat(aug_s)
    print(f"  Generated {len(aug_y)} synthetic sequences for classes {fine_grained_classes}")
    return aug_X, aug_y, aug_s

# ── Cell 9 | DANN Training ───────────────────────────────────────
def train_main_model(F_ext, clf, dom_clf, train_dl, cfg, device):
    params = (list(F_ext.parameters()) +
              list(clf.parameters()) +
              list(dom_clf.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg['lr_main'], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs_main'])
    ce  = nn.CrossEntropyLoss()

    F_ext.train(); clf.train(); dom_clf.train()
    train_losses = []

    for epoch in range(1, cfg['epochs_main'] + 1):
        lam = dann_schedule(epoch, cfg['epochs_main'])
        ep_loss, nb = 0., 0

        for batch_x, batch_y, batch_s in train_dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_s = batch_s.to(device)

            feat       = F_ext(batch_x)
            loss_cls   = ce(clf(feat), batch_y)

            real_mask  = (batch_s >= 0)
            if real_mask.sum() > 1:
                loss_dom = ce(dom_clf(feat[real_mask], lam=lam), batch_s[real_mask])
            else:
                loss_dom = torch.tensor(0., device=device)

            loss = loss_cls + cfg['lambda_domain'] * loss_dom
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1

        sch.step()
        train_losses.append(ep_loss / nb)
        if epoch % 10 == 0:
            print(f"  Main epoch {epoch:3d}/{cfg['epochs_main']} | "
                  f"loss: {train_losses[-1]:.4f}  λ_dann: {lam:.3f}")
            
            # AUTOSAVE FEATURE: Triggers every 10 epochs
            torch.save({
                'feature_extractor': F_ext.state_dict(),
                'classifier': clf.state_dict(),
                'dom_clf': dom_clf.state_dict(),
                'epoch': epoch
            }, '/content/autosave_latest.pt')

    return F_ext, clf, train_losses

# ── Cell 10 | Evaluation ─────────────────────────────────────────
@torch.no_grad()
def evaluate(F_ext, clf, loader, device):
    F_ext.eval(); clf.eval()
    all_preds, all_labels = [], []
    for batch_x, batch_y, _ in loader:
        preds = clf(F_ext(batch_x.to(device))).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())
    return np.array(all_preds), np.array(all_labels)

def print_results(preds, labels, fold_id, cfg):
    overall_f1 = f1_score(labels, preds, average='macro')
    fg_mask    = np.isin(labels, cfg['fine_grained'])
    fg_f1      = f1_score(labels[fg_mask], preds[fg_mask], average='macro')
    print(f"\n{'─'*55}")
    print(f"  Fold {fold_id} Results")
    print(f"{'─'*55}")
    print(f"  Overall   macro-F1 : {overall_f1:.4f}")
    print(f"  Fine-grained F1    : {fg_f1:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=GESTURE_NAMES)}")
    return overall_f1, fg_f1

def plot_confusion_matrix(preds, labels, fold_id):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_NAMES,
                yticklabels=GESTURE_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — Fold {fold_id}')
    plt.tight_layout(); plt.show()

def plot_losses(g_losses, d_losses, train_losses, fold_id):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(g_losses, label='Generator',     color='steelblue')
    axes[0].plot(d_losses, label='Discriminator', color='tomato')
    axes[0].set_title(f'Fold {fold_id} — LSGAN Losses')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(train_losses, color='green')
    axes[1].set_title(f'Fold {fold_id} — DANN Training Loss')
    axes[1].set_xlabel('Epoch'); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

# ── Cell 11 | 2-Fold Cross-Validation ───────────────────────────
def make_subject_fold_indices(subjects, fold_id):
    unique_subjects = np.unique(subjects)
    S   = len(unique_subjects)    
    mid = S // 2                 
    
    # Dynamically split whatever subjects are present right down the middle
    if fold_id == 1:
        test_subjs  = set(unique_subjects[0:mid])
        train_subjs = set(unique_subjects[mid:S])
    else:
        test_subjs  = set(unique_subjects[mid:S])
        train_subjs = set(unique_subjects[0:mid])
        
    train_idx = np.where(np.isin(subjects, list(train_subjs)))[0]
    test_idx  = np.where(np.isin(subjects, list(test_subjs)))[0]
    return train_idx, test_idx

def run_experiment():
    X_np, y_np, s_np = load_dataset(DATA_DIR)
    full_dataset = SoliDataset(X_np, y_np, s_np)

    results = {'overall_f1': [], 'fg_f1': []}

    for fold in [1, 2]:
        print(f"\n{'='*55}")
        print(f"  FOLD {fold} / 2")
        print(f"{'='*55}")

        train_idx, test_idx = make_subject_fold_indices(s_np, fold)
        train_set = Subset(full_dataset, train_idx)
        test_set  = Subset(full_dataset, test_idx)

        train_dl = DataLoader(train_set, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True)
        test_dl  = DataLoader(test_set,  batch_size=CFG['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

        print(f"\n[Stage 1] Training Fast LSGAN  (epochs={CFG['epochs_gan']}) ...")
        G     = FastSequenceGenerator(CFG['latent_dim'], CFG['num_classes'],
                                      CFG['T'], CFG['R'], CFG['D']).to(device)
        D_gan = FastSequenceDiscriminator(CFG['num_classes'],
                                          CFG['T'], CFG['R'], CFG['D']).to(device)
        G, g_losses, d_losses = train_lsgan(G, D_gan, train_dl, CFG, device)

        print(f"\n[Stage 2] Generating synthetic sequences ...")
        aug_X, aug_y, aug_s = generate_augmented_sequences(
            G, CFG['fine_grained'], CFG['n_aug'], CFG, device)

        combined_dl = DataLoader(
            ConcatDataset([train_set,
                           AugWrapper(TensorDataset(aug_X, aug_y, aug_s))]),
            batch_size=CFG['batch_size'], shuffle=True,
            num_workers=2, pin_memory=True)

        print(f"\n[Stage 3] Training DANN classifier  (epochs={CFG['epochs_main']}) ...")
        F_ext   = FeatureExtractor(CFG['feat_dim']).to(device)
        clf     = GestureClassifier(CFG['feat_dim'], CFG['num_classes']).to(device)
        dom_clf = DomainClassifier(CFG['feat_dim'], CFG['num_subjects']).to(device)

        F_ext, clf, train_losses = train_main_model(
            F_ext, clf, dom_clf, combined_dl, CFG, device)

        preds, labels = evaluate(F_ext, clf, test_dl, device)
        ovr, fg = print_results(preds, labels, fold, CFG)
        results['overall_f1'].append(ovr)
        results['fg_f1'].append(fg)

        plot_confusion_matrix(preds, labels, fold)
        plot_losses(g_losses, d_losses, train_losses, fold)

        ckpt_path = f'/content/checkpoint_fold{fold}.pt'
        torch.save({
            'feature_extractor': F_ext.state_dict(),
            'classifier'       : clf.state_dict(),
            'generator'        : G.state_dict(),
            'fold'             : fold,
            'cfg'              : CFG,
        }, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")

    print(f"\n{'='*55}")
    print("  FINAL SUMMARY  (avg over 2 folds)")
    print(f"{'='*55}")
    print(f"  Overall macro-F1 : {np.mean(results['overall_f1']):.4f} "
          f"± {np.std(results['overall_f1']):.4f}")
    print(f"  Fine-grained F1  : {np.mean(results['fg_f1']):.4f} "
          f"± {np.std(results['fg_f1']):.4f}")
    return results

# ── Cell 12 | RUN ────────────────────────────────────────────────
# results = run_experiment()

# ── Cell 13 | Ablation Study ─────────────────────────────────────
# (You can run this manually after the main loop finishes)
def run_ablation(mode='baseline', fold=1):
    X_np, y_np, s_np = load_dataset(DATA_DIR)
    full_dataset = SoliDataset(X_np, y_np, s_np)
    train_idx, test_idx = make_subject_fold_indices(s_np, fold)
    train_set = Subset(full_dataset, train_idx)
    test_set  = Subset(full_dataset, test_idx)

    train_dl = DataLoader(train_set, batch_size=CFG['batch_size'],
                          shuffle=True, num_workers=2)
    test_dl  = DataLoader(test_set,  batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=2)

    use_gan  = mode in ('gan_only',  'full')
    use_dann = mode in ('dann_only', 'full')

    if use_gan:
        G     = FastSequenceGenerator(CFG['latent_dim'], CFG['num_classes'],
                                      CFG['T'], CFG['R'], CFG['D']).to(device)
        D_gan = FastSequenceDiscriminator(CFG['num_classes'],
                                          CFG['T'], CFG['R'], CFG['D']).to(device)
        G, _, _ = train_lsgan(G, D_gan, train_dl, CFG, device)
        aug_X, aug_y, aug_s = generate_augmented_sequences(
            G, CFG['fine_grained'], CFG['n_aug'], CFG, device)
        combined_dl = DataLoader(
            ConcatDataset([train_set,
                           AugWrapper(TensorDataset(aug_X, aug_y, aug_s))]),
            batch_size=CFG['batch_size'], shuffle=True, num_workers=2)
    else:
        combined_dl = train_dl

    cfg_abl = dict(CFG)
    if not use_dann:
        cfg_abl['lambda_domain'] = 0.0

    F_ext   = FeatureExtractor(CFG['feat_dim']).to(device)
    clf     = GestureClassifier(CFG['feat_dim'], CFG['num_classes']).to(device)
    dom_clf = DomainClassifier(CFG['feat_dim'], CFG['num_subjects']).to(device)
    F_ext, clf, _ = train_main_model(F_ext, clf, dom_clf, combined_dl, cfg_abl, device)

    preds, labels = evaluate(F_ext, clf, test_dl, device)
    ovr = f1_score(labels, preds, average='macro')
    fg  = f1_score(labels[np.isin(labels, CFG['fine_grained'])],
                   preds [np.isin(labels, CFG['fine_grained'])],
                   average='macro')
    
    # ---------------------------------------------------------
    # NEW CODE: Print the detailed classification report
    # ---------------------------------------------------------
    gesture_names = ['Finger Slider', 'Finger Rub', 'Pinch Index', 'Pinch Pinky', 
                     'Tap/Click', 'Full Pinch', 'Push Wave', 'Pull Wave', 
                     'Palm Hold', 'Side Tap', 'Air CW']
    
    print(f"\n======================================================")
    print(f"   Classification Report for Ablation Mode: '{mode}'  ")
    print(f"======================================================")
    print(classification_report(labels, preds, target_names=gesture_names, zero_division=0))
    print(f"[{mode:12s}]  Overall Macro F1={ovr:.4f}   Fine-grained F1={fg:.4f}\n")
    # ---------------------------------------------------------

    plot_confusion_matrix(preds, labels, fold)

    return ovr, fg

for m in ['baseline', 'gan_only', 'dann_only', 'full']:
    run_ablation(mode=m, fold=2)
