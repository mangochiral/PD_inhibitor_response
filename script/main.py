import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import shap
import umap
from  makemeta import Makemetadat
from modelutils import MultiModalDataset, MultiModalNet
import torch.optim as optim

# Working directory
print("Enter the data directory: ")
data_directory = input()
os.join(data_directory)

# Making metadata
# file path
if __name__ == "__main__":
    file_path = os.path.join(data_directory, "/GSE91061_series_matrix.txt")
    metadata_df = Makemetadat(file_path)
    metadata_df.head()
    metadata_df.to_csv(os.path.join(data_directory, "/metadata_GSE.csv"))

    # Cytosolic DNA sensing scores
    cyto = pd.read_csv(os.path.join(data_directory, "/GSE91061_BMS038109Sample_Cytolytic_Score_20161026.txt"), sep="\t")
    cyto.columns = ['title', 'CytoScore']

    # Getting the gene symbol for geneid
    anto = pd.read_csv(os.path.join(data_directory, "/Human.GRCh38.p13.annot.tsv"), sep='\t')
    anto = anto[['GeneID', 'Symbol']]
    # Getting the raw counts data
    raw_counts = pd.read_csv(os.path.join(data_directory, "/GSE91061_raw_counts_GRCh38.p13_NCBI.tsv"), sep="\t")
    raw_counts = pd.merge(raw_counts, anto, on='GeneID')

    #Filtering data and setting up for data Normalisation
    # Remove geneID column
    raw_counts = raw_counts.iloc[:, 1:]
    raw_counts.set_index('Symbol', inplace=True)
    # Drop rows where the raw counts of genes is less than 10 and is present in less than 10 samples
    filtered_df = raw_counts[(raw_counts >= 10).sum(axis=1) >= 10]

    # Log normalization
    log_df = np.log1p(filtered_df)
    log_df = log_df.fillna(0)

    # View result
    metadata_df = pd.read_csv(os.path.join(data_directory ,"/metadata_GSE.csv"), index_col=0)
    # Add the cytosolic scores to the metadata
    metadata_df = pd.merge(metadata_df, cyto, on='title')

    #Make Response to treatment and time point binary
    metadata_df["response"] = metadata_df["response"].map({"PRCR": 1, "SD": 0, "PD": 0, "UNK": "UNK"})
    metadata_df = metadata_df[metadata_df['response'] != "UNK"]
    metadata_df["visit"] = metadata_df["visit"].map({"Pre": 0, "On": 1})

    # Set the Accession Number of the patient samples as index of metadata
    metadata_df.set_index("geo_accession", inplace=True)
    metadata_df = metadata_df[["visit", "CytoScore", "response"]]

    # Make separate pd.Series for predicting the response 
    labels_mat = metadata_df['response']
    labels_mat.head()

    # Remove the samples column not present in metadata transpose log counts data 
    # for running torch
    columns_match = metadata_df.index
    log_df_filtered = log_df[columns_match]
    log_df_filtered.head()
    # Transpose log counts data
    trans_log_filtered = log_df_filtered.T
    trans_log_filtered.head()

    # Removing the response column from metadata to use it as clinical signature
    metadata_df = metadata_df[["visit", "CytoScore"]]
    metadata_df.head()

    # Prepare dataset and loader
    dataset = MultiModalDataset(trans_log_filtered, metadata_df, labels_mat)

    train_frac = 0.8
    n_train = int(train_frac * len(dataset))
    n_test  = len(dataset) - n_train

    # ensure reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_test],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    # Instantiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalNet(trans_log_filtered.shape[1], metadata_df.shape[1])

    # Set up loss function and optimizer 
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train & evaluate 
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        # Training phase 
        model.train()
        train_loss = 0.0
        for rna_b, clin_b, lab_b in train_loader:
            optimizer.zero_grad()
            preds = model(rna_b, clin_b).squeeze()
            loss  = criterion(preds, lab_b.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * rna_b.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch:02d} — Train Loss: {train_loss:.4f}")

    #  Evaluation phase 
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for rna_b, clin_b, lab_b in test_loader:
                p = model(rna_b, clin_b).squeeze()
                all_preds.append(p.cpu())
                all_labels.append(lab_b.squeeze().cpu())
        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Optional: ROC-AUC if scikit-learn is installed
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
            print(f" Test  AUC:  {auc:.4f}")
        except ImportError:
            pass
        # Save
    torch.save(model.state_dict(), 'model.pth')
    print("Training & evaluation complete!")

    #Ground Truth response vs prediction
    df = pd.DataFrame()
    df['ground_truth'] = all_labels.numpy()
    df['prediction'] = all_preds.numpy()

    # Explainability
    # Use first sample as background
    bg_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    bg_rna, bg_clin, _ = next(iter(bg_loader))

    # Move to device
    bg_rna = bg_rna.to(device)
    bg_clin = bg_clin.to(device)
    explainer = shap.GradientExplainer(model, [bg_rna, bg_clin])

    # Compute SHAP
    shap_rna, shap_clin = [], []
    for i, (rna_s, clin_s, _) in enumerate(DataLoader(dataset, batch_size=1)):
        val_r, val_c = rna_s.to(device), clin_s.to(device)
        vals = explainer.shap_values([val_r, val_c])
        shap_rna.append(vals[0]); shap_clin.append(vals[1])
    shap_rna = np.vstack(shap_rna).squeeze(-1); shap_clin = np.vstack(shap_clin).squeeze(-1)

    # Align and plot
    ids = dataset.ids
    rna_vals = trans_log_filtered.loc[ids].values
    clin_vals = metadata_df.loc[ids].values

    print("Paste plot path")
    plot_path = input()
    # Shap plot for Gene signature
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_rna, rna_vals, feature_names=trans_log_filtered.columns, show=False);
    plt.tight_layout()
    # Save the plot to a file
    plt.savefig(os.path.join(plot_path,"/shap_rna.png")) 

    # Display the plot after saving
    plt.show() 

    # Shap plot for clinical signature
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_clin, clin_vals, feature_names=metadata_df.columns, show=False)
    plt.tight_layout()
    # Save the plot to a file
    plt.savefig(os.path.join(plot_path,"shap_clinical.png"))

    # Display the plot after saving
    plt.show() 

    #----UMAP----
    # Prepare your RNA tensor (n_samples × n_genes)
    X_rna = torch.tensor(trans_log_filtered.values, dtype=torch.float32).to(device)

    # Compute the embedding
    with torch.no_grad():
        rna_embeddings = model.rna_branch(X_rna)
    # Assuming `rna_emb_np` (n_samples × 64) and `labels` (n_samples,) are in scope

    # 1) UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(rna_embeddings)

    # 2) Scatter plot
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_mat, alpha=0.7)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("RNA-Branch Embeddings UMAP")
    plt.colorbar(scatter, label="Response")
    plt.tight_layout()
    # Save the plot to a file
    plt.savefig(os.path.join(plot_path,"umap_response.png"))

    # Display the plot after saving
    plt.show() 
