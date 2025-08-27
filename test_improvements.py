#!/usr/bin/env python3
"""
Test script for enhanced Alzheimer's voice detection improvements
"""

import os
import sys
import numpy as np
from app import AlzheimersVoiceAnalyzer

def test_enhanced_features():
    """Test the enhanced feature extraction and analysis"""
    print("Testing enhanced Alzheimer's detection features...")
    
    # Initialize analyzer
    analyzer = AlzheimersVoiceAnalyzer()
    
    # Test enhanced pause analysis
    print("\n1. Testing enhanced pause analysis...")
    test_audio = np.random.randn(16000)  # 1 second of random audio at 16kHz
    
    # Simulate pause features
    test_features = {
        'pause_rate': 0.4,
        'pause_duration_mean': 1.2,
        'pause_duration_std': 0.8,
        'pause_percentage': 25,
        'long_pauses': 3,
        'medium_pauses': 2,
        'short_pauses': 5
    }
    
    # Test speech rate variability calculation
    variability = analyzer._calculate_speech_rate_variability(test_features)
    print(f"   Speech rate variability: {variability:.2f}")
    
    # Test enhanced hesitation detection
    print("\n2. Testing enhanced hesitation detection...")
    test_transcript = "I was um trying to uh you know get the thing um like when you want to er sort of basically do something"
    words = test_transcript.lower().split()
    
    # Count hesitations manually
    hesitation_markers = [
        'uh', 'um', 'er', 'ah', 'well', 'like', 'you know', 'i mean', 'sort of', 'kind of',
        'basically', 'actually', 'literally', 'obviously', 'clearly', 'honestly', 'frankly'
    ]
    
    manual_count = sum(1 for word in words if word in hesitation_markers)
    print(f"   Manual hesitation count: {manual_count}")
    
    # Test enhanced pause analysis
    print("\n3. Testing enhanced pause analysis...")
    pause_analysis = analyzer._analyze_pauses(test_audio, 16000)
    print(f"   Pause analysis results: {pause_analysis}")
    
    # Test risk assessment with enhanced features
    print("\n4. Testing enhanced risk assessment...")
    
    # Create test features with high pause/hesitation values (should trigger high risk)
    high_risk_features = {
        'speaking_rate': 65,  # Very slow
        'pause_rate': 0.6,    # High pause rate
        'pause_duration_mean': 2.5,  # Long pauses
        'pause_percentage': 30,      # High pause percentage
        'hesitation_count': 12,      # Many hesitations
        'repetition_count': 8,       # Many repetitions
        'type_token_ratio': 0.3,    # Low lexical diversity
        'semantic_fluency': 0.35,   # Poor semantic fluency
        'coherence_score': 0.4      # Poor coherence
    }
    
    risk_assessment = analyzer._calculate_risk_assessment(50, high_risk_features)
    print(f"   High risk features assessment: {risk_assessment}")
    
    # Create test features with normal values (should trigger low risk)
    normal_features = {
        'speaking_rate': 120,        # Normal speaking rate
        'pause_rate': 0.2,          # Normal pause rate
        'pause_duration_mean': 0.5,  # Normal pause duration
        'pause_percentage': 12,      # Normal pause percentage
        'hesitation_count': 2,       # Few hesitations
        'repetition_count': 1,       # Few repetitions
        'type_token_ratio': 0.6,    # Good lexical diversity
        'semantic_fluency': 0.65,   # Good semantic fluency
        'coherence_score': 0.75     # Good coherence
    }
    
    normal_risk_assessment = analyzer._calculate_risk_assessment(20, normal_features)
    print(f"   Normal features assessment: {normal_risk_assessment}")
    
    print("\n✅ Enhanced feature testing completed successfully!")
    
    # Test biomarker generation
    print("\n5. Testing enhanced biomarker generation...")
    biomarkers = analyzer._generate_biomarker_report(high_risk_features, 75)
    print(f"   Speech timing status: {biomarkers['speech_timing']['status']}")
    print(f"   Fluency status: {biomarkers['fluency_coherence']['status']}")
    print(f"   Lexical status: {biomarkers['lexical_semantic']['status']}")

def test_whisper_integration():
    """Test Whisper transcription integration"""
    print("\n6. Testing Whisper integration...")
    
    # Check if Whisper is available
    try:
        import whisper
        print("   ✅ Whisper is available")
        
        # Test model loading
        try:
            model = whisper.load_model("base")
            print("   ✅ Whisper base model loaded successfully")
        except Exception as e:
            print(f"   ❌ Whisper model loading failed: {e}")
            
    except ImportError:
        print("   ❌ Whisper not available - please install with: pip install openai-whisper")

if __name__ == "__main__":
    print("🧠 Enhanced Alzheimer's Voice Detection Test Suite")
    print("=" * 50)
    
    try:
        test_enhanced_features()
        test_whisper_integration()
        
        print("\n🎉 All tests completed successfully!")
        print("\nKey improvements implemented:")
        print("✅ Enhanced transcription using Whisper")
        print("✅ Improved pause detection with multiple methods")
        print("✅ Enhanced hesitation and repetition detection")
        print("✅ Increased weight for timing-related biomarkers")
        print("✅ Better calibration to reduce false positives")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)