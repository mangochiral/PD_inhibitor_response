# === 1. Define your Dataset ===
class MultiModalDataset(Dataset):
    def __init__(self, rna_df, clin_df, label_df):
        # convert pandas DataFrames → torch tensors
        self.rna = torch.from_numpy(rna_df.to_numpy()).float()
        clin_mat = clin_df.apply(pd.to_numeric, errors='coerce')
        self.clinical = torch.from_numpy(clin_mat.to_numpy()).float()
        lab_mat = label_df.apply(pd.to_numeric, errors='coerce')
        self.labels = torch.from_numpy(lab_mat.to_numpy()).float()
        # sanity check
        self.ids = rna_df.index.intersection(clin_df.index).intersection(label_df.index)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.rna[idx], self.clinical[idx], self.labels[idx]
    
# === 5. Define your model ===
class MultiModalNet(nn.Module):
    def __init__(self, rna_dim, clin_dim):
        super().__init__()
        self.rna_branch = nn.Sequential(nn.Linear(rna_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),nn.Dropout(0.4), 
                                        nn.Linear(512, 256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,64))
        self.clin_branch = nn.Sequential(nn.Linear(clin_dim, 32),nn.ReLU(), nn.Dropout(0.3), nn.Linear(32,16))
        self.classifier = nn.Sequential(nn.Linear(80, 32),nn.ReLU(),nn.Linear(32, 1),nn.Sigmoid())

    def forward(self, rna, clin):
        x1 = self.rna_branch(rna)
        x2 = self.clin_branch(clin)
        x  = torch.cat([x1, x2], dim=1)
        return self.classifier(x)