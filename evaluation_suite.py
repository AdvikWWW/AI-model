import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import librosa
import os
from app import AlzheimersVoiceAnalyzer
import warnings
warnings.filterwarnings('ignore')

class AlzheimersEvaluationSuite:
    def __init__(self):
        self.analyzer = AlzheimersVoiceAnalyzer()
        self.feature_domains = {
            'temporal': ['speaking_rate', 'pause_rate', 'pause_duration_mean', 'pause_duration_std', 
                        'speech_rate_variability', 'articulation_rate'],
            'voice_quality': ['jitter', 'shimmer', 'voice_breaks', 'voiceless_ratio'],
            'linguistic': ['type_token_ratio', 'hesitation_count', 'repetition_count', 
                          'semantic_fluency', 'syntactic_complexity', 'coherence_score'],
            'prosodic': ['f0_mean', 'f0_std', 'intensity_mean', 'intensity_std'],
            'enhanced': ['word_length_variability', 'function_word_ratio', 'content_word_density',
                        'idea_density', 'propositional_density', 'narrative_coherence',
                        'phonetic_complexity', 'syllable_complexity', 'cognitive_load_index']
        }
        
    def create_synthetic_dataset(self, n_subjects=100, recordings_per_subject=3):
        """Create synthetic dataset with subject IDs for proper train/test split"""
        np.random.seed(42)
        
        data = []
        for subject_id in range(n_subjects):
            # Assign subject to AD or control (40% AD prevalence)
            is_ad = np.random.random() < 0.4
            
            for recording_id in range(recordings_per_subject):
                # Generate correlated features within subject
                base_noise = np.random.normal(0, 0.1, len(self.analyzer.feature_names))
                
                if is_ad:
                    # AD subjects have more impaired speech patterns
                    features = {
                        'speaking_rate': max(np.random.normal(85, 15), 40),
                        'pause_rate': min(np.random.normal(0.45, 0.15), 0.8),
                        'pause_duration_mean': max(np.random.normal(1.2, 0.4), 0.3),
                        'pause_duration_std': max(np.random.normal(0.8, 0.3), 0.1),
                        'jitter': max(np.random.normal(0.025, 0.01), 0.005),
                        'shimmer': max(np.random.normal(0.08, 0.03), 0.02),
                        'voice_breaks': max(np.random.poisson(4), 0),
                        'voiceless_ratio': min(np.random.normal(0.4, 0.1), 0.7),
                        'type_token_ratio': max(np.random.normal(0.35, 0.1), 0.1),
                        'hesitation_count': max(np.random.poisson(8), 0),
                        'repetition_count': max(np.random.poisson(5), 0),
                        'semantic_fluency': max(np.random.normal(0.45, 0.15), 0.1),
                        'syntactic_complexity': max(np.random.normal(6, 2), 2),
                        'coherence_score': max(np.random.normal(0.5, 0.2), 0.1),
                        'f0_mean': max(np.random.normal(180, 30), 80),
                        'f0_std': max(np.random.normal(25, 10), 5),
                        'intensity_mean': max(np.random.normal(65, 10), 40),
                        'intensity_std': max(np.random.normal(8, 3), 2)
                    }
                else:
                    # Control subjects have normal speech patterns
                    features = {
                        'speaking_rate': max(np.random.normal(140, 20), 80),
                        'pause_rate': min(np.random.normal(0.25, 0.1), 0.5),
                        'pause_duration_mean': max(np.random.normal(0.6, 0.2), 0.2),
                        'pause_duration_std': max(np.random.normal(0.4, 0.2), 0.1),
                        'jitter': max(np.random.normal(0.012, 0.005), 0.005),
                        'shimmer': max(np.random.normal(0.045, 0.015), 0.02),
                        'voice_breaks': max(np.random.poisson(1), 0),
                        'voiceless_ratio': min(np.random.normal(0.25, 0.08), 0.5),
                        'type_token_ratio': max(np.random.normal(0.55, 0.1), 0.3),
                        'hesitation_count': max(np.random.poisson(3), 0),
                        'repetition_count': max(np.random.poisson(2), 0),
                        'semantic_fluency': max(np.random.normal(0.7, 0.1), 0.4),
                        'syntactic_complexity': max(np.random.normal(10, 3), 4),
                        'coherence_score': max(np.random.normal(0.75, 0.15), 0.4),
                        'f0_mean': max(np.random.normal(200, 25), 100),
                        'f0_std': max(np.random.normal(20, 8), 5),
                        'intensity_mean': max(np.random.normal(70, 8), 50),
                        'intensity_std': max(np.random.normal(6, 2), 2)
                    }
                
                # Add enhanced features
                enhanced_features = {
                    'speech_rate_variability': max(np.random.normal(1.5 if is_ad else 1.0, 0.5), 0.5),
                    'articulation_rate': features['speaking_rate'] * (1 - features['pause_rate']),
                    'word_length_variability': max(np.random.normal(2.0, 0.5), 0.5),
                    'function_word_ratio': min(np.random.normal(0.4, 0.1), 0.7),
                    'content_word_density': max(np.random.normal(0.6, 0.1), 0.3),
                    'idea_density': max(np.random.normal(0.4 if is_ad else 0.6, 0.1), 0.1),
                    'propositional_density': max(np.random.normal(0.3 if is_ad else 0.5, 0.1), 0.1),
                    'narrative_coherence': max(np.random.normal(0.5 if is_ad else 0.8, 0.2), 0.1),
                    'phonetic_complexity': max(np.random.normal(0.8, 0.2), 0.2),
                    'syllable_complexity': max(np.random.normal(2.2, 0.4), 1.0),
                    'cognitive_load_index': max(np.random.normal(8 if is_ad else 4, 3), 0),
                    'signal_quality': max(np.random.normal(0.8, 0.1), 0.3),
                    'noise_robustness_score': max(np.random.normal(0.85, 0.1), 0.4)
                }
                
                features.update(enhanced_features)
                
                # Ensure all feature names are present
                feature_vector = []
                for fname in self.analyzer.feature_names:
                    if fname in features:
                        feature_vector.append(features[fname])
                    else:
                        feature_vector.append(np.random.normal(0.5, 0.1))
                
                data.append({
                    'subject_id': subject_id,
                    'recording_id': f"{subject_id}_{recording_id}",
                    'features': np.array(feature_vector),
                    'label': int(is_ad)
                })
        
        return pd.DataFrame(data)
    
    def subject_level_split(self, data, test_size=0.3, random_state=42):
        """Perform subject-level train/test split to prevent data leakage"""
        unique_subjects = data['subject_id'].unique()
        
        # Use GroupShuffleSplit to ensure no subject appears in both train and test
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(data, groups=data['subject_id']))
        
        train_data = data.iloc[train_idx].reset_index(drop=True)
        test_data = data.iloc[test_idx].reset_index(drop=True)
        
        print(f"Train subjects: {len(train_data['subject_id'].unique())}")
        print(f"Test subjects: {len(test_data['subject_id'].unique())}")
        print(f"Subject overlap: {len(set(train_data['subject_id'].unique()) & set(test_data['subject_id'].unique()))}")
        
        return train_data, test_data
    
    def bootstrap_metrics(self, y_true, y_pred, y_prob, n_bootstrap=1000, confidence_level=0.95):
        """Compute evaluation metrics with confidence intervals using bootstrapping"""
        np.random.seed(42)
        
        metrics = {
            'roc_auc': [],
            'pr_auc': [],
            'sensitivity': [],
            'specificity': [],
            'f1': []
        }
        
        n_samples = len(y_true)
        alpha = 1 - confidence_level
        
        for i in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices]
            
            # Skip if bootstrap sample has only one class
            if len(np.unique(y_true_boot)) < 2:
                continue
            
            # Compute metrics
            try:
                metrics['roc_auc'].append(roc_auc_score(y_true_boot, y_prob_boot))
                metrics['pr_auc'].append(average_precision_score(y_true_boot, y_prob_boot))
                
                tn, fp, fn, tp = confusion_matrix(y_true_boot, y_pred_boot).ravel()
                metrics['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                metrics['f1'].append(f1_score(y_true_boot, y_pred_boot))
            except:
                continue
        
        # Compute confidence intervals
        results = {}
        for metric, values in metrics.items():
            if values:
                mean_val = np.mean(values)
                ci_lower = np.percentile(values, 100 * alpha/2)
                ci_upper = np.percentile(values, 100 * (1 - alpha/2))
                results[metric] = {
                    'mean': mean_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'std': np.std(values)
                }
        
        return results
    
    def ablation_study(self, train_data, test_data):
        """Perform ablation study by removing each feature domain"""
        print("Starting ablation study...")
        
        # Baseline: all features
        X_train = np.vstack(train_data['features'].values)
        y_train = train_data['label'].values
        X_test = np.vstack(test_data['features'].values)
        y_test = test_data['label'].values
        
        # Train baseline model
        self.analyzer.scaler.fit(X_train)
        X_train_scaled = self.analyzer.scaler.transform(X_train)
        X_test_scaled = self.analyzer.scaler.transform(X_test)
        
        self.analyzer.model.fit(X_train_scaled, y_train)
        y_prob_baseline = self.analyzer.model.predict_proba(X_test_scaled)[:, 1]
        y_pred_baseline = self.analyzer.model.predict(X_test_scaled)
        
        baseline_metrics = self.bootstrap_metrics(y_test, y_pred_baseline, y_prob_baseline)
        
        results = {'baseline': baseline_metrics}
        
        # Test each domain removal
        for domain_name, feature_names in self.feature_domains.items():
            print(f"Testing without {domain_name} features...")
            
            # Find indices of features to remove
            remove_indices = []
            for fname in feature_names:
                if fname in self.analyzer.feature_names:
                    remove_indices.append(self.analyzer.feature_names.index(fname))
            
            if not remove_indices:
                continue
            
            # Create feature sets without this domain
            keep_indices = [i for i in range(len(self.analyzer.feature_names)) if i not in remove_indices]
            X_train_ablated = X_train[:, keep_indices]
            X_test_ablated = X_test[:, keep_indices]
            
            # Train model without this domain
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            
            scaler_ablated = StandardScaler()
            model_ablated = RandomForestClassifier(n_estimators=100, random_state=42)
            
            X_train_scaled_ablated = scaler_ablated.fit_transform(X_train_ablated)
            X_test_scaled_ablated = scaler_ablated.transform(X_test_ablated)
            
            model_ablated.fit(X_train_scaled_ablated, y_train)
            y_prob_ablated = model_ablated.predict_proba(X_test_scaled_ablated)[:, 1]
            y_pred_ablated = model_ablated.predict(X_test_scaled_ablated)
            
            ablated_metrics = self.bootstrap_metrics(y_test, y_pred_ablated, y_prob_ablated)
            results[f'without_{domain_name}'] = ablated_metrics
        
        return results
    
    def calibration_analysis(self, y_true, y_prob, n_bins=10):
        """Compute calibration curve and Brier score"""
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Brier score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Plot (Brier Score: {brier_score:.3f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/advikmishra/CascadeProjects/AlzheimersVoiceDetection/calibration_curve.png', dpi=300)
        plt.close()
        
        return {
            'brier_score': brier_score,
            'calibration_curve': (fraction_of_positives, mean_predicted_value)
        }
    
    def add_noise_to_audio(self, audio, snr_db):
        """Add white noise to audio at specified SNR"""
        if snr_db == np.inf:  # No noise
            return audio
        
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def noise_robustness_test(self, test_data, snr_levels=[np.inf, 10, 5, 0]):
        """Test model robustness to background noise"""
        print("Starting noise robustness test...")
        
        results = {}
        
        for snr_db in snr_levels:
            print(f"Testing at SNR: {snr_db} dB")
            
            predictions = []
            true_labels = []
            
            for idx, row in test_data.iterrows():
                # Simulate noisy audio analysis
                original_features = row['features']
                
                # Add noise effect to acoustic features (simplified simulation)
                if snr_db != np.inf:
                    noise_factor = 1 / (1 + 10**(snr_db/20))  # Noise degrades features
                    
                    # Apply noise to voice quality features more severely
                    noisy_features = original_features.copy()
                    
                    # Jitter and shimmer increase with noise
                    jitter_idx = self.analyzer.feature_names.index('jitter') if 'jitter' in self.analyzer.feature_names else None
                    shimmer_idx = self.analyzer.feature_names.index('shimmer') if 'shimmer' in self.analyzer.feature_names else None
                    
                    if jitter_idx is not None:
                        noisy_features[jitter_idx] *= (1 + noise_factor)
                    if shimmer_idx is not None:
                        noisy_features[shimmer_idx] *= (1 + noise_factor)
                    
                    # Voice breaks increase
                    vb_idx = self.analyzer.feature_names.index('voice_breaks') if 'voice_breaks' in self.analyzer.feature_names else None
                    if vb_idx is not None:
                        noisy_features[vb_idx] += noise_factor * 2
                    
                    features_to_use = noisy_features
                else:
                    features_to_use = original_features
                
                # Get prediction
                features_scaled = self.analyzer.scaler.transform(features_to_use.reshape(1, -1))
                prob = self.analyzer.model.predict_proba(features_scaled)[0, 1]
                
                predictions.append(prob)
                true_labels.append(row['label'])
            
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)
            pred_labels = (predictions > 0.5).astype(int)
            
            # Compute metrics
            metrics = self.bootstrap_metrics(true_labels, pred_labels, predictions, n_bootstrap=500)
            results[f'SNR_{snr_db}dB'] = metrics
        
        return results
    
    def plot_noise_robustness(self, noise_results):
        """Plot noise robustness results"""
        snr_levels = []
        roc_aucs = []
        roc_cis = []
        
        for key, metrics in noise_results.items():
            snr = key.replace('SNR_', '').replace('dB', '')
            if snr == 'inf':
                snr_val = 20  # For plotting purposes
            else:
                snr_val = float(snr)
            
            snr_levels.append(snr_val)
            roc_aucs.append(metrics['roc_auc']['mean'])
            ci_width = metrics['roc_auc']['ci_upper'] - metrics['roc_auc']['ci_lower']
            roc_cis.append(ci_width / 2)
        
        # Sort by SNR level
        sorted_data = sorted(zip(snr_levels, roc_aucs, roc_cis))
        snr_levels, roc_aucs, roc_cis = zip(*sorted_data)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(snr_levels, roc_aucs, yerr=roc_cis, marker='o', capsize=5, capthick=2)
        plt.xlabel('SNR (dB)')
        plt.ylabel('ROC-AUC')
        plt.title('Model Performance vs Background Noise Level')
        plt.grid(True, alpha=0.3)
        plt.xticks(snr_levels)
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        plt.savefig('/Users/advikmishra/CascadeProjects/AlzheimersVoiceDetection/noise_robustness.png', dpi=300)
        plt.close()
    
    def run_comprehensive_evaluation(self):
        """Run the complete evaluation suite"""
        print("=== Alzheimer's Voice Detection - Comprehensive Evaluation ===\n")
        
        # 1. Create synthetic dataset
        print("1. Creating synthetic dataset...")
        data = self.create_synthetic_dataset(n_subjects=100, recordings_per_subject=3)
        
        # 2. Subject-level split
        print("2. Performing subject-level train/test split...")
        train_data, test_data = self.subject_level_split(data)
        
        # 3. Train model
        print("3. Training model...")
        X_train = np.vstack(train_data['features'].values)
        y_train = train_data['label'].values
        X_test = np.vstack(test_data['features'].values)
        y_test = test_data['label'].values
        
        self.analyzer.scaler.fit(X_train)
        X_train_scaled = self.analyzer.scaler.transform(X_train)
        X_test_scaled = self.analyzer.scaler.transform(X_test)
        
        self.analyzer.model.fit(X_train_scaled, y_train)
        
        # 4. Get predictions
        y_prob = self.analyzer.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.analyzer.model.predict(X_test_scaled)
        
        # 5. Bootstrap metrics
        print("4. Computing metrics with confidence intervals...")
        metrics = self.bootstrap_metrics(y_test, y_pred, y_prob)
        
        # 6. Ablation study
        print("5. Running ablation study...")
        ablation_results = self.ablation_study(train_data, test_data)
        
        # 7. Calibration analysis
        print("6. Analyzing model calibration...")
        calibration_results = self.calibration_analysis(y_test, y_prob)
        
        # 8. Noise robustness test
        print("7. Testing noise robustness...")
        noise_results = self.noise_robustness_test(test_data)
        self.plot_noise_robustness(noise_results)
        
        # Print results
        self.print_results(metrics, ablation_results, calibration_results, noise_results)
        
        return {
            'metrics': metrics,
            'ablation': ablation_results,
            'calibration': calibration_results,
            'noise_robustness': noise_results
        }
    
    def print_results(self, metrics, ablation_results, calibration_results, noise_results):
        """Print comprehensive results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Main metrics
        print("\n📊 MAIN METRICS (95% CI)")
        print("-" * 40)
        for metric, values in metrics.items():
            print(f"{metric.upper():>12}: {values['mean']:.3f} [{values['ci_lower']:.3f}, {values['ci_upper']:.3f}]")
        
        # Ablation study
        print(f"\n🔬 ABLATION STUDY")
        print("-" * 40)
        baseline_auc = ablation_results['baseline']['roc_auc']['mean']
        print(f"{'Baseline (all features)':>25}: {baseline_auc:.3f}")
        
        for domain, results in ablation_results.items():
            if domain != 'baseline':
                auc = results['roc_auc']['mean']
                drop = baseline_auc - auc
                print(f"{domain:>25}: {auc:.3f} (Δ: {drop:+.3f})")
        
        # Calibration
        print(f"\n📏 CALIBRATION")
        print("-" * 40)
        print(f"Brier Score: {calibration_results['brier_score']:.3f}")
        
        # Noise robustness
        print(f"\n🔊 NOISE ROBUSTNESS")
        print("-" * 40)
        for condition, results in noise_results.items():
            auc = results['roc_auc']['mean']
            ci = f"[{results['roc_auc']['ci_lower']:.3f}, {results['roc_auc']['ci_upper']:.3f}]"
            print(f"{condition:>10}: {auc:.3f} {ci}")

if __name__ == "__main__":
    evaluator = AlzheimersEvaluationSuite()
    results = evaluator.run_comprehensive_evaluation()
