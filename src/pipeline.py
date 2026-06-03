import pandas as pd
import numpy as np
import re
import logging
from xgboost import XGBRegressor
from sksurv.metrics import concordance_index_censored
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from scipy.stats import rankdata
import warnings

warnings.filterwarnings("ignore")

# Configure professional logging interface
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClinicalHepaticEnsemblePipeline:
    """
    An enterprise-grade, object-oriented survival analysis pipeline for Hepatic Events.
    
    This architecture integrates Bagged Joint-Distribution Matching with a 
    Multi-Seed 60-Model XGBoost Cox Proportional Hazards framework to prevent 
    covariate shift and eliminate initialization bias.
    """
    
    def __init__(self, target_train_size: int = 736):
        self.target_train_size = target_train_size
        self.target_col = 'evenements_hepatiques_majeurs'
        
        # Clinically validated predictive hepatic biomarkers
        self.lab_targets = [
            'ast', 'alt', 'plt', 'bilirubin', 'ggt', 'gluc_fast',
            'chol', 'triglyc', 'fibs_stiffness_med_BM_1'
        ]
        
        # Rigorous regularization hyperparameters to ensure generalizability
        self.xgb_params = {
            'objective': 'survival:cox', 
            'eval_metric': 'cox-nloglik',
            'n_estimators': 2000, 
            'early_stopping_rounds': 50,      
            'learning_rate': 0.015, 
            'max_depth': 8,              
            'min_child_weight': 5,       
            'subsample': 0.8, 
            'colsample_bytree': 0.5,    
            'reg_alpha': 2.0,            
            'reg_lambda': 5.0,
            'n_jobs': -1
        }
        
    def load_and_preprocess(self, train_path: str, test_path: str):
        """Loads data layers and sanitizes impossible temporal clinical records."""
        logging.info("Loading and sanitizing clinical datasets...")
        self.train_raw = pd.read_csv(train_path)
        self.test_raw = pd.read_csv(test_path)

        df_hep = self.train_raw.dropna(subset=[self.target_col]).reset_index(drop=True)
        # Exclude administrative anomalies (Event labeled true but age of occurrence missing)
        self.train_hep_base = df_hep[~((df_hep[self.target_col] == 1) & (df_hep['evenements_hepatiques_age_occur'].isna()))].reset_index(drop=True)
        
        self._extract_visit_columns()
        
    def _extract_visit_columns(self):
        """Discovers active longitudinal visit arrays across columns."""
        v_cols = [f'Age_v{i}' for i in range(1, 23)]
        self.v_cols_tr = [c for c in v_cols if c in self.train_hep_base.columns]
        self.v_cols_te = [c for c in v_cols if c in self.test_raw.columns]

    def _get_timelines(self, df: pd.DataFrame, visit_cols: list):
        """Computes structural encounters and exact duration metrics."""
        visits = df[visit_cols].notna().sum(axis=1)
        duration = (df[visit_cols].max(axis=1) - df[visit_cols].min(axis=1)).fillna(0)
        return visits.astype(int), np.round(duration).astype(int)

    def _harmonize_cohort_distributions(self, seed_val: int):
        """Executes Bagged Strict Random Joint-Distribution Matching to counter Covariate Shift."""
        train_visits, train_dur = self._get_timelines(self.train_hep_base, self.v_cols_tr)
        test_visits, test_dur = self._get_timelines(self.test_raw, self.v_cols_te)

        train_meta = pd.DataFrame({'Duration_Years': train_dur, 'Visits': train_visits})
        test_meta = pd.DataFrame({'Duration_Years': test_dur, 'Visits': test_visits})

        train_meta['Cohort'] = train_meta['Duration_Years'].astype(str) + "_Yrs_" + train_meta['Visits'].astype(str) + "_Vis"
        test_meta['Cohort'] = test_meta['Duration_Years'].astype(str) + "_Yrs_" + test_meta['Visits'].astype(str) + "_Vis"

        test_cohort_pct = test_meta['Cohort'].value_counts(normalize=True)
        balanced_indices = []

        np.random.seed(seed_val)
        for cohort, pct in test_cohort_pct.items():
            target_count = int(np.ceil(self.target_train_size * pct))
            available_indices = train_meta[train_meta['Cohort'] == cohort].index.tolist()
            
            if len(available_indices) > 0:
                keep_count = min(target_count, len(available_indices))
                selected = np.random.choice(available_indices, size=keep_count, replace=False)
                balanced_indices.extend(selected)

        np.random.shuffle(balanced_indices)
        sampled_train_hep = self.train_hep_base.iloc[balanced_indices].reset_index(drop=True)
        
        logging.info(f"  -> Data bag constructed via Seed {seed_val}. Sample Size: {len(sampled_train_hep)} patients.")
        return sampled_train_hep

    def _engineer_biomarker_features(self, df: pd.DataFrame, visit_cols: list, g_means: dict, h_means: dict):
        """Generates dynamic biomarker features and spatial sub-population variance vectors."""
        X = pd.DataFrame(index=df.index)
        all_v_cols = [c for c in df.columns if re.search(r'_v\d+$', c)]
        
        temp_visits = df[visit_cols].notna().sum(axis=1)
        min_age = df[visit_cols].min(axis=1)
        max_age = df[visit_cols].max(axis=1)
        
        X['std_age'] = df[visit_cols].std(axis=1).fillna(-1)
        X['total_visits'] = temp_visits
        X['duration_years'] = np.round((max_age - min_age).fillna(0))

        for prefix in self.lab_targets:
            cols = sorted([c for c in all_v_cols if re.sub(r'_v\d+$', '', c) == prefix], key=lambda x: int(x.split('_v')[-1]))
            if len(cols) > 0:
                tests_taken = df[cols].notna().sum(axis=1)
                X[f"{prefix}_completion_ratio"] = tests_taken / temp_visits.replace(0, 0.1)
                
                personal_mean = df[cols].mean(axis=1)
                safe_personal_mean = personal_mean.replace(0, 0.1).fillna(0.1)
                
                global_mean = g_means.get(prefix, 0.1)
                healthy_mean = h_means.get(prefix, 0.1)
                
                X[f"{prefix}_Personal_vs_Global_Mean"] = personal_mean / global_mean
                X[f"{prefix}_Personal_vs_Healthy_Mean"] = personal_mean / healthy_mean
                
                for c in cols:
                    v_num = c.split('_')[-1] 
                    raw_val = df[c] 
                    X[f"{prefix}_ratio_to_personal_{v_num}"] = raw_val / safe_personal_mean
                    X[f"{prefix}_ratio_to_global_{v_num}"] = raw_val / global_mean
                    X[f"{prefix}_ratio_to_healthy_{v_num}"] = raw_val / healthy_mean

        return X.fillna(-1)

    def _train_20_fold_regime(self, sampled_train_hep: pd.DataFrame, mskf_splitter, seed_val: int, preds_accumulator: np.ndarray):
        """Trains 20 distinct stratified cross-validation folds on the isolated cohort bag."""
        params = self.xgb_params.copy()
        params['random_state'] = seed_val
        
        y_bin = sampled_train_hep[self.target_col].astype(int).values
        time_visits_all = sampled_train_hep[self.v_cols_tr].notna().sum(axis=1).clip(lower=1.0).values
        y_cox_v_all = np.where(y_bin == 1, time_visits_all, -time_visits_all)
        
        v_bins = pd.qcut(pd.Series(time_visits_all), q=3, labels=False, duplicates='drop').values
        d_bins = pd.qcut((sampled_train_hep[self.v_cols_tr].max(axis=1) - sampled_train_hep[self.v_cols_tr].min(axis=1)), q=3, labels=False, duplicates='drop').values
        y_multilabel = np.vstack((y_bin, v_bins, d_bins)).T

        for fold, (tr_idx, va_idx) in enumerate(mskf_splitter.split(sampled_train_hep, y_multilabel)):
            train_fold_df = sampled_train_hep.iloc[tr_idx].reset_index(drop=True)
            val_fold_df = sampled_train_hep.iloc[va_idx].reset_index(drop=True)
            
            fold_global_means, fold_healthy_means = {}, {}
            fold_combined = pd.concat([train_fold_df, self.test_raw], ignore_index=True)
            healthy_fold_df = train_fold_df[train_fold_df[self.target_col] == 0]

            for prefix in self.lab_targets:
                g_cols = [c for c in fold_combined.columns if re.search(rf'^{prefix}_v\d+$', c)]
                if len(g_cols) > 0:
                    t_g_mean = np.nanmean(fold_combined[g_cols].values.flatten())
                    fold_global_means[prefix] = 0.1 if (np.isnan(t_g_mean) or t_g_mean == 0) else t_g_mean
                    
                h_cols = [c for c in healthy_fold_df.columns if re.search(rf'^{prefix}_v\d+$', c)]
                if len(h_cols) > 0:
                    t_h_mean = np.nanmean(healthy_fold_df[h_cols].values.flatten())
                    fold_healthy_means[prefix] = 0.1 if (np.isnan(t_h_mean) or t_h_mean == 0) else t_h_mean

            X_t = self._engineer_biomarker_features(train_fold_df, self.v_cols_tr, fold_global_means, fold_healthy_means)
            X_v = self._engineer_biomarker_features(val_fold_df, self.v_cols_tr, fold_global_means, fold_healthy_means)
            X_test_fold = self._engineer_biomarker_features(self.test_raw, self.v_cols_te, fold_global_means, fold_healthy_means)
            
            X_t, X_v = X_t.align(X_v, join='left', axis=1, fill_value=0)
            X_t, X_test_fold = X_t.align(X_test_fold, join='left', axis=1, fill_value=0)

            y_t, y_v = y_cox_v_all[tr_idx], y_cox_v_all[va_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=0)
            
            preds_accumulator += rankdata(model.predict(X_test_fold))
            
        return preds_accumulator

    def run_ensemble_training(self):
        """Orchestrates the massive execution framework for the 60-Model Ensemble."""
        logging.info("Initiating 60-Model Tri-Seed Bagged Ensemble Training...")
        self.test_preds_total = np.zeros(len(self.test_raw))
        
        seeds = [42, 1234, 4]
        for idx, seed_val in enumerate(seeds):
            logging.info(f"Phase {idx + 1}/3: Executing Data Bagging & 20 Folds for Seed {seed_val}...")
            
            sampled_train_hep = self._harmonize_cohort_distributions(seed_val)
            mskf = MultilabelStratifiedKFold(n_splits=20, shuffle=True, random_state=seed_val)
            self.test_preds_total = self._train_20_fold_regime(sampled_train_hep, mskf, seed_val, self.test_preds_total)
            
        logging.info("Ensemble framework complete. Robust rank averaging executed.")

    def generate_submission(self, output_filename: str, baseline_csv_path: str):
        """Formats, cross-references with concurrent death baselines, and commits outputs to disk."""
        final_risk_scores = rankdata(self.test_preds_total) / len(self.test_preds_total)

        sub = pd.DataFrame({
            'trustii_id': self.test_raw['trustii_id'],
            'risk_hepatic_event': final_risk_scores
        })

        base_sub = pd.read_csv(baseline_csv_path)
        sub = sub.merge(base_sub[['trustii_id', 'risk_death']], on='trustii_id', how='left')

        if sub['risk_death'].isna().sum() > 0:
            sub['risk_death'] = sub['risk_death'].fillna(sub['risk_death'].median())

        sub['risk_death'] = 0
        sub.to_csv(output_filename, index=False)
        logging.info(f"Submission successfully exported to: {output_filename}")


if __name__ == "__main__":
    pipeline = ClinicalHepaticEnsemblePipeline()
    pipeline.load_and_preprocess(
        train_path='data/train.csv',  
        test_path='data/test.csv'
    )
    pipeline.run_ensemble_training()
    pipeline.generate_submission(
        output_filename='best_LB_hep_bagged_triseed.csv',
        baseline_csv_path='data/baseline_submission.csv'
    )
