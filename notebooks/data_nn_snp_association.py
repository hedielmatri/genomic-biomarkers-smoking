import sys
#!pip install XXX --target ./my_custom_packages
sys.path.append("./my_custom_packages")


import polars as pl
import torch
import numpy as np
from collections import Counter
import time
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def load_and_filter_genotypes(genotype_path, annot_path, min_maf):
    """
    Loads data
    Aligns both datasets
    MAF per SNP
    Removes monomorphic and rare variants
    """
    
    annot_df = pl.read_parquet(annot_path).filter(pl.col("Status") != "Ex-smoker")
    genotype_df = pl.read_parquet(genotype_path)
    
    # Individual IDs
    ind_cols = [str(row) for row in annot_df["Sample"].to_list()]
    
    # SNP column
    snp_col = genotype_df.columns[0]
    
    # Only SNP column + columns with patients 
    kept_cols = [snp_col] + [col for col in ind_cols if col in genotype_df.columns]
    genotype_df = genotype_df.select(kept_cols)
    
    print(f"Current shape: {genotype_df.height} SNPs with {len(kept_cols) - 1} indviduals")
    
    # Get all ind columns
    updated_ind_cols = [col for col in genotype_df.columns if col != snp_col]
    
    # MAF and Monomorphic filtering
    # Combine all genotype strings into one sequence string
    seq_col = pl.concat_str([pl.col(c) for c in updated_ind_cols])
    
    # Count each nucleotide for that snp on all individuals
    cnt_a = seq_col.str.count_matches("A")
    cnt_c = seq_col.str.count_matches("C")
    cnt_g = seq_col.str.count_matches("G")
    cnt_t = seq_col.str.count_matches("T")
    
    # Total valid alleles in row
    total_alleles = cnt_a + cnt_c + cnt_g + cnt_t
    
    # Srt descending and grab minor allele count
    counts_list = pl.concat_list([cnt_a, cnt_c, cnt_g, cnt_t])
    minor_allele_count = counts_list.list.sort(descending = True).list.get(1)
    
    # MAF per SNP
    maf_snp = minor_allele_count / total_alleles
    
    # MAF and Monomorphic filtering
    filtered_df = genotype_df.filter(maf_snp >= min_maf)
    
    print(f"Filtered shape: {filtered_df.height} SNPs remaining")
    
    return filtered_df, annot_df



def build_one_hot_tensor(filtered_df, snp_col):
    """
    Transpose genotype and encode into a one hot tensor
    """
    # Individuals IDs 
    ind_ids = [col for col in filtered_df.columns if col != snp_col]
    n_inds = len(ind_ids)
    n_snps = filtered_df.height
        
    # Create matrix in RAM
    state_matrix = np.full((n_inds, n_snps), fill_value = -1, dtype = np.int8)
    
    # genotype only as a list of series
    genotype_matrix = filtered_df.select(ind_ids).to_numpy()
    

    # Mapping allele string to numbers for faster and more efficient transpose
    for snp_idx in range(n_snps):
        row_genotypes = genotype_matrix[snp_idx]
        
        # Count frequency of each string in row
        counts = Counter([g for g in row_genotypes])

        # Most common = 0, 2nd = 1, 3rd = 2
        ordered_alleles = [pair[0] for pair in counts.most_common(3)]
        mapping = {allele: n for n, allele in enumerate(ordered_alleles)}
        
        # Fill the patient state array for this SNP
        for inds_idx, allele in enumerate(row_genotypes):
            if allele in mapping:
                state_matrix[inds_idx, snp_idx] = mapping[allele]
                
    # Convert to tensor
    state_tensor = torch.from_numpy(state_matrix).to(torch.int64)
    
    # missing values to 0 for one_hot
    missing = state_tensor == -1
    state_tensor[missing] = 0
    
    # (N, K) -> (N, K, 3)
    one_hot_3d = torch.nn.functional.one_hot(state_tensor, num_classes = 3).to(torch.float16)
    
    # Set missing values ([0,0,0]) to zero across all 3 bits
    one_hot_3d[missing] = 0.0
    
    # Flatten to 2D -> (n_inds, K_snps * 3)
    tensor = one_hot_3d.view(n_inds, -1)
    
    print(f"One Hot tensor shape of {tensor.shape} with memory of {tensor.element_size() * tensor.nelement() / 1e9:.2f} gb")
    
    return tensor, ind_ids


def extract_ancestry_and_covariates(tensor, annot_df, inds_ids, target_variance):
    """
    Calculates total variance of genotype matrix
    SVD and to find the minimum number of sv to explain a target_variance

    Returns a covariate tensor of shape (N_patients, k_pcs + 2).
    """
    print(f"SVD components to reach {target_variance * 100:.1f}% explained variance")
    
    n_patients, n_features = tensor.shape

    # float16 tensor to float32 for SVD
    x = tensor.to(torch.float32) 
    
    # Center the matrix (subtract mean per SNP)
    x_centered = x - x.mean(dim = 0, keepdim = True)

    # Gram matrix across all features
    gram = torch.matmul(x_centered, x_centered.T)

    gram_np = gram.cpu().numpy().astype(np.float64)

    print("Running eigendecomposition")
    eigenvalues_np, eigenvectors_np = np.linalg.eigh(gram_np)    

    eigenvalues = torch.from_numpy(eigenvalues_np).to(torch.float32)
    eigenvectors = torch.from_numpy(eigenvectors_np).to(torch.float32)

    # descending order 
    eigenvalues = torch.flip(eigenvalues, dims = [0])
    eigenvectors = torch.flip(eigenvectors, dims = [1])
    eigenvalues = torch.clamp(eigenvalues, min = 0.0)
    
    # Variance breakdown
    total_variance = torch.sum(eigenvalues).item()
    var_explained_per_pc = (eigenvalues / total_variance).cpu().numpy()
    cummu_var_explained = np.cumsum(var_explained_per_pc)
    
    # Find exact k where cumulative variance hits target_variance
    k_needed = int(np.searchsorted(cummu_var_explained, target_variance) + 1)
    actual_var = cummu_var_explained[k_needed - 1] * 100
    
    print(f"{n_features} features:")
    print(f"Total variance: {total_variance:.2f}")
    print(f"Top PC1 explains {var_explained_per_pc[0]*100:.2f}%")
    print(f"Top PC2 explains {var_explained_per_pc[1]*100:.2f}%")
    print(f"Top PC3 explains: {var_explained_per_pc[2]*100:.2f}%")
    print(f"Top {k_needed} PCs explain {actual_var:.2f}% of total variance")
    
    # Project Top k PCs with U_k * sqrt(lambda_k)
    U_k = eigenvectors[:, : k_needed]
    S_k = torch.sqrt(eigenvalues[ : k_needed])
    # Shape -> (3399, k_needed)
    ancestry_pcs = U_k * S_k.unsqueeze(0)
    
    # Add Age, Gender
    annot_aligned = annot_df.filter(pl.col("Sample").cast(pl.Utf8).is_in(inds_ids))
    
    raw_age = annot_aligned["Age"].to_numpy().astype(np.float32)
    # Z score normalisation
    small_number = 1e-4
    age_normalized = (raw_age - raw_age.mean()) / (raw_age.std() + small_number)
    
    raw_gender = annot_aligned["Gender"].str.to_lowercase().to_numpy()
    gender_binary = (raw_gender == "male").astype(np.float32)
    
    covariates = np.column_stack([age_normalized, gender_binary])
    tensor_cov = torch.from_numpy(covariates).to(torch.float32)
    
    # Covariate Tensor with Top k ancestry PCs + Age + Gender
    covariates_tensor = torch.cat([ancestry_pcs, tensor_cov], dim = 1).to(torch.float16)
    
    print(f"Covariate Tensor Shape: {covariates_tensor.shape}")
    
    return covariates_tensor, cummu_var_explained

def get_stratified_folds(y_tensor, n_splits):
    """
    Stratified K-Fold  on target Y
    Returns list of (train_indices, val_indices).
    """
    y_np = y_tensor.squeeze().cpu().numpy()
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 4)
    
    folds = []
    for train_idx, test_idx in skf.split(np.zeros(len(y_np)), y_np):
        folds.append((train_idx, test_idx))
        
    print(f"{n_splits} Stratified K-Fold with train {len(folds[0][0])} and test {len(folds[0][1])}")
    return folds


def feature_selection(x_train, y_train, n_permutations, percentile):
    """
    feature selection threshold by establishing the maximum correlation by random
    Keeps features that beat random floor
    """

    x_f32 = x_train.to(torch.float32)
    y_f32 = y_train.to(torch.float32)
    
    x_centered = x_f32 - x_f32.mean(dim = 0, keepdim = True)
    y_centered = y_f32 - y_f32.mean()

    small_number = 1e-4
    x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=0) + small_number)
    
    # Calculate correlations
    y_std_real = torch.sqrt(torch.sum(y_centered ** 2) + small_number)
    covariances_real = torch.matmul(y_centered.T, x_centered).squeeze(0)
    real_correlations = torch.abs(covariances_real / (x_std * y_std_real))
    
    # Random floor

    max_random_correlation = 0.0
    max_noises = []
    for _ in range(n_permutations):
        
        # Shuffle target
        shuffled_indices = torch.randperm(y_f32.shape[0])
        y_shuffled = y_f32[shuffled_indices]
        
        y_shuf_centered = y_shuffled - y_shuffled.mean()
        y_std_shuf = torch.sqrt(torch.sum(y_shuf_centered ** 2) + small_number)
        
        # Calculate correlations
        covariances_shuf = torch.matmul(y_shuf_centered.T, x_centered).squeeze(0)
        noise_correlations = torch.abs(covariances_shuf / (x_std * y_std_shuf))
        
        current_max_noise = torch.max(noise_correlations).item()
        
        max_noises.append(current_max_noise)

        if current_max_noise > max_random_correlation:
            max_random_correlation = current_max_noise
            
    ## Filtering
    # Keep features that score higher
    #selected = real_correlations > max_random_correlation
    selected = real_correlations > max_random_correlation * 1.50
    #selected = real_correlations > np.percentile(max_noises, percentile)

    #selected_indices = torch.nonzero(selected).squeeze()

    ''' 
    real_corr_grouped = real_correlations.view(-1, 3)
    selected_snps = torch.any(real_corr_grouped > (max_random_correlation * 1.02), dim = 1)
    selected = selected_snps.unsqueeze(1).expand(-1, 3).reshape(-1)
    '''
    selected_indices = torch.nonzero(selected).squeeze()
    
    if selected_indices.numel() == 0:
        print("No feature selected")
        return None
    
    n_kept = selected_indices.numel() if selected_indices.dim() > 0 else 1
    total_features = x_train.shape[1]
    
    print(f"Max correlation by random is {max_random_correlation:.4f}")
    print(f"Kept {n_kept} of {total_features} features")
    
    return selected_indices


class efn(nn.Module):
    def __init__(self, n_genetic_features, n_covariates, hidden_dim1, hidden_dim2):
        super(efn, self).__init__()
        
        # Feature selector, learnable vector put to 1.0 
        #self.feature_selector = nn.Parameter(torch.ones(n_genetic_features))
        
        # Epistatic Extraction Layers
        self.genetics_branch = nn.Sequential(
            nn.Linear(n_genetic_features, hidden_dim1),
            #nn.LayerNorm(hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.SiLU(), # Swish activation: smooth, nonlinear, no dead neurons
            nn.Dropout(0.5), # dropout to not rely on very few SNPs
            
            nn.Linear(hidden_dim1, hidden_dim2),
            #nn.LayerNorm(hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        #self.covariate_norm = nn.BatchNorm1d(n_covariates)
        '''
        self.linear_shortcut = nn.Sequential(
            nn.Linear(n_genetic_features, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2)
        )
        '''        
        # Concatenate the genetic output (hidden_dim2) with covariates and classify
        self.classifier = nn.Sequential(
         #   nn.Linear(hidden_dim2 + n_covariates, 32),
            nn.Linear(hidden_dim2 + n_covariates, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Outputs logits
        )

        self.linear_shortcut = nn.Linear(n_genetic_features, 1)

        self.linear_covariates = nn.Linear(n_covariates, 1)


    def forward(self, x_genetic, x_covariates):
        # Pass features through feature selector
        #kept_features = x_genetic * self.feature_selector
        
        #cov_norm = self.covariate_norm(x_covariates)

        # Take interactions
        genetic_exp = self.genetics_branch(x_genetic)

        #linear_exp = self.linear_shortcut(x_genetic)
        
        # Merge with covariates
        #final_exp = torch.cat([genetic_exp, cov_norm], dim = 1)

        final_exp = torch.cat([genetic_exp, x_covariates], dim = 1)
        
        # classification
        logits = self.classifier(final_exp)
        
        linear_logits = self.linear_shortcut(x_genetic)

        linear_cov = self.linear_covariates(x_covariates)

        return logits + linear_logits + linear_cov
    
    def get_gate_l1_loss(self):
        # return the L1 norm of feature selector
        return torch.sum(torch.abs(self.feature_selector))


def train_single_fold(fold_num, x_train, y_train, cov_train, x_test, y_test, cov_test, initial_lr, l1_lambda, stopping_limit, lr_limit, weight_decay):
    """
    Trains EpistaticFusionNet by gradient descent
    """
    n_genetics = x_train.shape[1]
    n_covars = cov_train.shape[1]
    
    # Initialize network
    model = efn(n_genetic_features = n_genetics, n_covariates = n_covars, hidden_dim1 = 128, hidden_dim2 = 32)
    
    optimiser = torch.optim.AdamW(model.parameters(), lr = initial_lr, weight_decay = weight_decay)
    
    # Update LR based on AUC
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode = "max", factor = 0.5, patience = lr_limit, min_lr = 0.0001
    )

    '''
    small_number =  1e-4
    num_neg, num_pos = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight_val = (num_neg / (num_pos + small_number)).to(torch.float32)
    loss_function = nn.BCEWithLogitsLoss(pos_weight = pos_weight_val)
    '''

    loss_function = nn.BCEWithLogitsLoss()
    
    best_test_auc = 0.0
    best_test_pred = None
    limit_counter = 0
    epoch = 1
    
    while True:
        model.train()
        optimiser.zero_grad()
        
        logits = model(x_train, cov_train)
        
        
        bce_loss = loss_function(logits, y_train)
        # l1_loss = l1_lambda * model.get_gate_l1_loss()
        loss = bce_loss # + l1_loss
        
        loss.backward()
        optimiser.step()
        
        model.eval()
        with torch.no_grad():
            test_logits = model(x_test, cov_test)
            test_prob = torch.sigmoid(test_logits).cpu().numpy().squeeze()
            
            y_test_np = y_test.cpu().numpy().squeeze()
            test_auc = roc_auc_score(y_test_np, test_prob)
            
        scheduler.step(test_auc)
        
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_pred = test_prob
            limit_counter = 0
        else:
            limit_counter += 1
            
        if limit_counter >= stopping_limit:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Fold {fold_num} stopped at epoch {epoch} with best test AUC of {best_test_auc:.4f} and LR of {current_lr:.2e}")
            break
        
        if epoch % 10 == 0:
            print(f"Currently on epoch {epoch} of fold {fold_num} with curr bestauc of {best_test_auc:.4f}")
        epoch += 1
            
    return best_test_auc, best_test_pred


def run_pipeline_test(genotype_path, annot_path, min_maf, target_variance):
    """
    Test pipeline
    """
    start_time = time.time()
    print("Starting...")

    t0 = time.time()
    filtered_df, annot_df = load_and_filter_genotypes(genotype_path, annot_path, min_maf)
    snp_col_name = filtered_df.columns[0]
    t1 = time.time()
    print(f"load_and_filter_genotypes took: {t1 - t0:.2f}s")

    t2 = time.time()
    tensor, inds_ids = build_one_hot_tensor(filtered_df, snp_col_name)
    t3 = time.time()
    print(f"build_one_hot_tensor_parallel took: {t3 - t2:.2f}s")

    annot_aligned = annot_df.filter(pl.col("Sample").cast(pl.Utf8).is_in(inds_ids))
    y = (annot_aligned["Status"].str.to_lowercase() == "smoker").to_numpy().astype(np.float32)
    y_tensor = torch.from_numpy(y).unsqueeze(1).to(torch.float16)

    t4 = time.time()
    covariates_tensor, cum_var = extract_ancestry_and_covariates(
        tensor, annot_df, inds_ids, target_variance
    )
    t5 = time.time()
    print(f"extract_ancestry_and_covariates took: {t5 - t4:.2f}s")

    t6 = time.time()
    folds = get_stratified_folds(y_tensor, n_splits = 5)
    
    folds_predictions = np.zeros(len(y_tensor), dtype = np.float32)
    folds_targets = y_tensor.squeeze().cpu().numpy().astype(np.float32)
    folds_aucs = []

    for fold_num, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"Fold {fold_num} out of 5")
        
        x_train = tensor[train_idx]
        y_train = y_tensor[train_idx]
        cov_train = covariates_tensor[train_idx].to(torch.float32)
        
        x_test = tensor[test_idx]
        y_test = y_tensor[test_idx]
        cov_test = covariates_tensor[test_idx].to(torch.float32)
        
        # Feature selection
        selected_indices = feature_selection(x_train, y_train, n_permutations = 50, percentile = 95)
        
        x_train_sel = x_train[:, selected_indices].to(torch.float32)
        x_test_sel = x_test[:, selected_indices].to(torch.float32)
        
        # Train model
        best_auc, test_preds = train_single_fold(
            fold_num = fold_num, 
            x_train = x_train_sel, y_train = y_train.to(torch.float32), cov_train = cov_train, 
            x_test = x_test_sel, y_test = y_test.to(torch.float32), cov_test = cov_test,
            initial_lr = 0.005, stopping_limit = 100, lr_limit = 25, weight_decay = 0.0001, l1_lambda = 0.00001
        )
        
        # Track predictions for overall score
        folds_predictions[test_idx] = test_preds
        folds_aucs.append(best_auc)

    # Out of fold scores
    total_folds_auc = roc_auc_score(folds_targets, folds_predictions)
    total_folds_pr_auc = average_precision_score(folds_targets, folds_predictions)
    folds_binary_preds = (folds_predictions >= 0.5).astype(np.float32)
    total_folds_acc = accuracy_score(folds_targets, folds_binary_preds)
    
    t7 = time.time()
    
    print(f"Mean fold ROC AUC: {np.mean(folds_aucs):.4f}")
    print(f"Total ROC AUC: {total_folds_auc:.4f}")
    print(f"Total PR AUC: {total_folds_pr_auc:.4f}")
    print(f"Total accuracy: {total_folds_acc * 100:.2f}%")

    print(f"Training took: {t7 - t6:.2f}s")


    elapsed = time.time() - start_time
    print()
    print(f"Took {elapsed:.2f}s")

    return tensor, y_tensor, covariates_tensor, inds_ids
    
x, y, cov, inds_ids = run_pipeline_test(
    genotype_path = "GSE148375_processed_data.parquet", 
    annot_path = "annot_data_processed.parquet", 
    min_maf  = 0.01,
    target_variance = 0.15
)

'''
First try: initial_lr = 0.005, l1_lambda = 0.0001, stopping_limit = 30, lr_limit = 10, weight_decay = 0.01 with hidden_dim1 = 256, hidden_dim2 = 32

Mean fold ROC AUC: 0.6159
Total ROC AUC: 0.6090
Total PR AUC: 0.5916
Total accuracy: 58.66%
Training took: 11.62s

Second try: hidden_dim1 = 512, hidden_dim2 = 128

Mean fold ROC AUC: 0.6139
Total ROC AUC: 0.6125
Total PR AUC: 0.6174
Total accuracy: 58.42%
Training took: 14.60s


Third try: initial_lr = 0.02, stopping_limit = 60, lr_limit = 15, weight_decay = 0.01 with hidden_dim1 = 128, hidden_dim2 = 32
Remove L1 layer LayerNorm, no more feature selector, added BatchNorm1d instead

Mean fold ROC AUC: 0.6237
Total ROC AUC: 0.6221
Total PR AUC: 0.6216
Total accuracy: 58.27%
Training took: 13.25s

Took 148.15s


Fourth attempt: initial_lr = 0.005
Added a linear pathway for the model, if linear (additive) is better, changed the linear classifier to account for it

Mean fold ROC AUC: 0.6303
Total ROC AUC: 0.6304
Total PR AUC: 0.6240
Total accuracy: 58.89%
Training took: 14.48s

Took 151.45s

Fifth attempt: same
target_variance changed from 15% to 5%, changing the permutation score to median of the maximums of the 50 permutations, dropout lowered 0.3 and 0.1 (down from 0.6 and 0.4)

Mean fold ROC AUC: 0.6075
Total ROC AUC: 0.5941
Total PR AUC: 0.6106
Total accuracy: 57.86%
Training took: 12.09s

Took 146.91s


Sixth attempt: lr = 0.001, weight_decay = 0.05
target_variance brought back up to 15%, keeping best of 50 permutations no median (reverted), dropout goes up to 0.5 and 0.2, giving it 64 dims to work with (up from 32) so hidden dim2 bumped to 64
adding a min_lr of 0.0001 to scheduler 


Mean fold ROC AUC: 0.6259
Total ROC AUC: 0.6184
Total PR AUC: 0.6031
Total accuracy: 58.89%
Training took: 14.49s

Took 155.37s

Seventh attempt: hidden_dim1 = 256
using 95 percentile of the best permutation, added a cov variant normaliser, changed linear to end of program and added to the total logit directly (old lines are commented)

Mean fold ROC AUC: 0.6204
Total ROC AUC: 0.6183
Total PR AUC: 0.6198
Total accuracy: 58.16%
Training took: 12.35s

Took 146.92s

Eighth attempt: 
Added class balance weight "pos_weight" to loss function, removed covariate_norm (reverted), added linear covariate andincluded in final logit, reusing max percentile score.
Mean fold ROC AUC: 0.6189
Total ROC AUC: 0.6159
Total PR AUC: 0.6148
Total accuracy: 58.24%
Training took: 15.82s

Took 149.77s


Nineth attempt: hidden_dim1 = 128, hidden_dim2 = 32
Adding a 5% strictness to random noise floor, standard bce loss (reverted)
Mean fold ROC AUC: 0.6223
Total ROC AUC: 0.6138
Total PR AUC: 0.6069
Total accuracy: 57.27%
Training took: 17.82s

Took 148.52s


10th attempt: initial_lr = 0.002, stopping_limit = 100, lr_limit = 25, weight_decay = 0.01 (much more time to hit a floor)
Decreassing noise floor margin to 2%, if any snp pass correlation test keeping all three of hot encoded snps instead of only the passing third,
Mean fold ROC AUC: 0.6059
Total ROC AUC: 0.6024
Total PR AUC: 0.5888
Total accuracy: 57.39%
Training took: 17.74s

Took 154.33s

11th attempt: initial_lr = 0.002, stopping_limit = 100, lr_limit = 25, weight_decay = 0.05
only keeping whatever passes the threshold (reverted), increasing floor margin to 15%
Mean fold ROC AUC: 0.6273
Total ROC AUC: 0.6249
Total PR AUC: 0.6224
Total accuracy: 59.45%
Training took: 16.82s

Took 155.28s

12th attempt: initial_lr = 0.005, weight_decay = 0.0001
increasing floor margin to 50%
Mean fold ROC AUC: 0.6246
Total ROC AUC: 0.6170
Total PR AUC: 0.6029
Total accuracy: 58.72%
Training took: 17.71s

Took 150.01s
'''
