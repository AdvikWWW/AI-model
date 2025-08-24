import os
import requests
import zipfile
import tarfile
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import time
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class AlzheimersDatasetDownloader:
    """
    Download and preprocess public Alzheimer's speech datasets
    """
    
    def __init__(self, base_dir="./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'adress_2020': {
                'name': 'ADReSS Challenge 2020',
                'url': 'https://archive.org/download/ADReSS-IS2020-data/ADReSS-IS2020-data.zip',
                'description': 'Balanced dataset with 78 AD + 78 HC participants',
                'format': 'zip',
                'audio_format': 'wav',
                'transcripts': True
            },
            'adresso_2021': {
                'name': 'ADReSSo Challenge 2021', 
                'url': 'https://archive.org/download/ADReSSo-IS2021-data/ADReSSo-IS2021-data.zip',
                'description': 'Audio-only dataset with 87 AD + 79 HC participants',
                'format': 'zip',
                'audio_format': 'wav',
                'transcripts': False
            },
            'dementia_bank_sample': {
                'name': 'DementiaBank Sample',
                'url': 'https://media.talkbank.org/dementia/English/Pitt/cookie.zip',
                'description': 'Sample from Pitt corpus Cookie Theft descriptions',
                'format': 'zip',
                'audio_format': 'wav',
                'transcripts': True
            }
        }
        
        self.feature_extractor = None
    
    def download_dataset(self, dataset_key, force_redownload=False):
        """Download a specific dataset"""
        
        if dataset_key not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        dataset_info = self.datasets[dataset_key]
        dataset_dir = self.base_dir / dataset_key
        
        if dataset_dir.exists() and not force_redownload:
            print(f"✅ {dataset_info['name']} already exists at {dataset_dir}")
            return dataset_dir
        
        print(f"📥 Downloading {dataset_info['name']}...")
        print(f"   Description: {dataset_info['description']}")
        
        # Create dataset directory
        dataset_dir.mkdir(exist_ok=True)
        
        # Download file
        try:
            response = requests.get(dataset_info['url'], stream=True)
            response.raise_for_status()
            
            filename = urlparse(dataset_info['url']).path.split('/')[-1]
            filepath = dataset_dir / filename
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Downloaded to {filepath}")
            
            # Extract if compressed
            if dataset_info['format'] == 'zip':
                print("📦 Extracting ZIP archive...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                os.remove(filepath)  # Remove zip after extraction
                
            elif dataset_info['format'] == 'tar.gz':
                print("📦 Extracting TAR.GZ archive...")
                with tarfile.open(filepath, 'r:gz') as tar_ref:
                    tar_ref.extractall(dataset_dir)
                os.remove(filepath)  # Remove tar after extraction
            
            print(f"✅ {dataset_info['name']} ready at {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_info['name']}: {e}")
            return None
    
    def create_synthetic_adress_data(self):
        """Create synthetic data matching ADReSS format for demonstration"""
        
        print("🔧 Creating synthetic ADReSS-format data for demonstration...")
        
        synthetic_dir = self.base_dir / "synthetic_adress"
        synthetic_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        train_dir = synthetic_dir / "train"
        test_dir = synthetic_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Generate synthetic audio and metadata
        np.random.seed(42)
        
        datasets = {
            'train': {'ad': 60, 'control': 60},
            'test': {'ad': 18, 'control': 18}
        }
        
        all_metadata = []
        
        for split, counts in datasets.items():
            split_dir = synthetic_dir / split
            
            for label, count in counts.items():
                label_dir = split_dir / label
                label_dir.mkdir(exist_ok=True)
                
                for i in range(count):
                    # Generate synthetic audio (Cookie Theft description simulation)
                    duration = np.random.uniform(30, 90)  # 30-90 seconds
                    sample_rate = 16000
                    
                    # Create realistic speech-like audio
                    t = np.linspace(0, duration, int(duration * sample_rate))
                    
                    # Base speech formants
                    f1, f2, f3 = 500, 1500, 2500  # Typical vowel formants
                    
                    if label == 'ad':
                        # AD characteristics: more pauses, slower speech, tremor
                        speech_rate = np.random.uniform(0.6, 0.8)  # Slower
                        pause_prob = 0.3  # More pauses
                        tremor_freq = np.random.uniform(4, 8)  # Voice tremor
                        jitter = np.random.uniform(0.02, 0.05)  # Higher jitter
                    else:
                        # Control characteristics: normal speech patterns
                        speech_rate = np.random.uniform(0.8, 1.0)  # Normal rate
                        pause_prob = 0.15  # Fewer pauses
                        tremor_freq = 0  # No tremor
                        jitter = np.random.uniform(0.005, 0.015)  # Lower jitter
                    
                    # Generate speech-like signal
                    audio = np.zeros_like(t)
                    
                    # Add speech segments with pauses
                    segment_length = 0.5  # 500ms segments
                    for seg_start in np.arange(0, duration, segment_length):
                        if np.random.random() > pause_prob:  # Not a pause
                            seg_end = min(seg_start + segment_length * speech_rate, duration)
                            seg_indices = (t >= seg_start) & (t < seg_end)
                            
                            # Generate formant-based speech
                            seg_audio = (np.sin(2 * np.pi * f1 * t[seg_indices]) * 0.3 +
                                       np.sin(2 * np.pi * f2 * t[seg_indices]) * 0.2 +
                                       np.sin(2 * np.pi * f3 * t[seg_indices]) * 0.1)
                            
                            # Add jitter and tremor
                            if tremor_freq > 0:
                                tremor = np.sin(2 * np.pi * tremor_freq * t[seg_indices]) * 0.1
                                seg_audio *= (1 + tremor)
                            
                            # Add jitter (frequency modulation)
                            jitter_mod = 1 + jitter * np.random.randn(len(seg_audio)) * 0.1
                            seg_audio *= jitter_mod
                            
                            audio[seg_indices] = seg_audio
                    
                    # Add realistic noise
                    noise_level = 0.05 if label == 'control' else 0.08
                    audio += np.random.randn(len(audio)) * noise_level
                    
                    # Normalize
                    audio = audio / np.max(np.abs(audio)) * 0.8
                    
                    # Save audio file
                    filename = f"{split}_{label}_{i:03d}.wav"
                    filepath = label_dir / filename
                    sf.write(filepath, audio, sample_rate)
                    
                    # Create metadata
                    metadata = {
                        'file_id': f"{split}_{label}_{i:03d}",
                        'file_path': str(filepath.relative_to(synthetic_dir)),
                        'label': 1 if label == 'ad' else 0,
                        'diagnosis': 'AD' if label == 'ad' else 'Control',
                        'split': split,
                        'duration': duration,
                        'sample_rate': sample_rate,
                        'synthetic': True
                    }
                    all_metadata.append(metadata)
        
        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(synthetic_dir / "metadata.csv", index=False)
        
        print(f"✅ Created synthetic dataset with {len(all_metadata)} samples")
        print(f"   Train: {len(metadata_df[metadata_df['split'] == 'train'])} samples")
        print(f"   Test: {len(metadata_df[metadata_df['split'] == 'test'])} samples")
        print(f"   AD: {len(metadata_df[metadata_df['label'] == 1])} samples")
        print(f"   Control: {len(metadata_df[metadata_df['label'] == 0])} samples")
        
        return synthetic_dir
    
    def extract_features_from_dataset(self, dataset_dir, output_file=None):
        """Extract features from downloaded dataset"""
        
        print(f"🔍 Extracting features from {dataset_dir}...")
        
        # Import the analyzer for feature extraction
        import sys
        sys.path.append(str(Path(__file__).parent))
        from app import AlzheimersVoiceAnalyzer
        
        analyzer = AlzheimersVoiceAnalyzer()
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(list(dataset_dir.rglob(ext)))
        
        if not audio_files:
            print(f"❌ No audio files found in {dataset_dir}")
            return None
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        # Extract features from each file
        features_data = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                print(f"\r   Processing {i+1}/{len(audio_files)}: {audio_file.name}", end='', flush=True)
                
                # Determine label from file path
                path_str = str(audio_file).lower()
                if 'ad' in path_str or 'dementia' in path_str or 'alzheimer' in path_str:
                    label = 1
                elif 'control' in path_str or 'hc' in path_str or 'normal' in path_str:
                    label = 0
                else:
                    # Try to infer from parent directory
                    parent = audio_file.parent.name.lower()
                    if 'ad' in parent or 'dementia' in parent:
                        label = 1
                    elif 'control' in parent or 'hc' in parent:
                        label = 0
                    else:
                        label = -1  # Unknown
                
                # Extract acoustic features
                acoustic_features = analyzer.extract_acoustic_features(str(audio_file))
                
                # Create feature vector
                feature_row = {
                    'file_path': str(audio_file.relative_to(dataset_dir)),
                    'label': label,
                    **acoustic_features
                }
                
                features_data.append(feature_row)
                
            except Exception as e:
                print(f"\n⚠️  Error processing {audio_file}: {e}")
                continue
        
        print(f"\n✅ Extracted features from {len(features_data)} files")
        
        # Create DataFrame
        features_df = pd.DataFrame(features_data)
        
        # Remove unknown labels
        if -1 in features_df['label'].values:
            unknown_count = len(features_df[features_df['label'] == -1])
            features_df = features_df[features_df['label'] != -1]
            print(f"⚠️  Removed {unknown_count} files with unknown labels")
        
        # Save features
        if output_file is None:
            output_file = dataset_dir / "extracted_features.csv"
        
        features_df.to_csv(output_file, index=False)
        print(f"💾 Saved features to {output_file}")
        
        # Print summary
        ad_count = len(features_df[features_df['label'] == 1])
        control_count = len(features_df[features_df['label'] == 0])
        print(f"📊 Dataset summary:")
        print(f"   AD samples: {ad_count}")
        print(f"   Control samples: {control_count}")
        print(f"   Features: {len(features_df.columns) - 2}")  # Exclude file_path and label
        
        return features_df
    
    def download_all_available(self):
        """Download all available datasets"""
        
        print("🚀 Starting download of all available datasets...")
        
        downloaded_datasets = []
        
        # Try synthetic data first (always works)
        try:
            synthetic_dir = self.create_synthetic_adress_data()
            downloaded_datasets.append(('synthetic_adress', synthetic_dir))
        except Exception as e:
            print(f"❌ Error creating synthetic data: {e}")
        
        # Try real datasets (may fail due to access restrictions)
        for dataset_key in self.datasets.keys():
            try:
                dataset_dir = self.download_dataset(dataset_key)
                if dataset_dir:
                    downloaded_datasets.append((dataset_key, dataset_dir))
            except Exception as e:
                print(f"⚠️  Could not download {dataset_key}: {e}")
                print(f"   This is expected for restricted datasets")
        
        print(f"\n✅ Successfully prepared {len(downloaded_datasets)} datasets:")
        for name, path in downloaded_datasets:
            print(f"   - {name}: {path}")
        
        return downloaded_datasets

def main():
    """Main function to demonstrate dataset downloading"""
    
    print("🧠 ALZHEIMER'S DATASET DOWNLOADER")
    print("=" * 50)
    
    downloader = AlzheimersDatasetDownloader()
    
    # Download available datasets
    datasets = downloader.download_all_available()
    
    # Extract features from each dataset
    all_features = []
    
    for dataset_name, dataset_dir in datasets:
        print(f"\n🔍 Processing {dataset_name}...")
        
        try:
            features_df = downloader.extract_features_from_dataset(dataset_dir)
            if features_df is not None:
                features_df['dataset'] = dataset_name
                all_features.append(features_df)
        except Exception as e:
            print(f"❌ Error extracting features from {dataset_name}: {e}")
    
    # Combine all features
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Save combined dataset
        output_path = downloader.base_dir / "combined_features.csv"
        combined_features.to_csv(output_path, index=False)
        
        print(f"\n🎯 COMBINED DATASET SUMMARY:")
        print(f"   Total samples: {len(combined_features)}")
        print(f"   AD samples: {len(combined_features[combined_features['label'] == 1])}")
        print(f"   Control samples: {len(combined_features[combined_features['label'] == 0])}")
        print(f"   Features: {len(combined_features.columns) - 3}")  # Exclude metadata columns
        print(f"   Saved to: {output_path}")
        
        # Dataset breakdown
        print(f"\n📊 BY DATASET:")
        for dataset in combined_features['dataset'].unique():
            subset = combined_features[combined_features['dataset'] == dataset]
            ad_count = len(subset[subset['label'] == 1])
            control_count = len(subset[subset['label'] == 0])
            print(f"   {dataset}: {len(subset)} total ({ad_count} AD, {control_count} Control)")
        
        return combined_features
    else:
        print("❌ No features extracted from any dataset")
        return None

if __name__ == "__main__":
    features = main()
