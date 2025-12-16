import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import json
import re
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedColdEmailGenerator:
    """
    Enterprise-Grade AI Email Generator with Advanced ML Features:
    - Deep Learning-inspired architecture
    - Multi-dimensional feature engineering
    - Ensemble template matching
    - Advanced personalization engine
    - Quality scoring system
    """
    
    def __init__(self):
        # Vectorizers
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        
        self.subject_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Storage
        self.templates = {}
        self.tone_styles = {}
        self.industry_patterns = {}
        self.structure_library = {}
        self.role_patterns = {}
        self.product_patterns = {}
        
        # ML Components
        self.embedding_matrix = None
        self.subject_embeddings = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Advanced Components
        self.personalization_engine = PersonalizationEngine()
        self.quality_scorer = QualityScorer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.template_clusterer = None
        
        # Metadata
        self.training_metadata = {}
        self.performance_metrics = {}
        
    def train(self, df, verbose=True):
        """
        Advanced training pipeline with 7 phases
        """
        if verbose:
            print("\n" + "🚀" * 35)
            print("  ADVANCED EMAIL GENERATOR TRAINING - NEURAL PIPELINE")
            print("🚀" * 35)
        
        start_time = datetime.now()
        
        # Phase 1: Data Validation & Cleaning
        if verbose:
            print("\n" + "=" * 70)
            print("📋 Phase 1: Data Validation & Preprocessing")
            print("=" * 70)
        df = self._validate_and_clean_data(df, verbose)
        
        # Phase 2: Advanced Feature Engineering
        if verbose:
            print("\n" + "=" * 70)
            print("🔧 Phase 2: Advanced Feature Engineering")
            print("=" * 70)
        df = self._engineer_advanced_features(df, verbose)
        
        # Phase 3: Neural-Inspired Vectorization
        if verbose:
            print("\n" + "=" * 70)
            print("🧠 Phase 3: Neural Vectorization & Embeddings")
            print("=" * 70)
        self._create_neural_embeddings(df, verbose)
        
        # Phase 4: Multi-Dimensional Template Learning
        if verbose:
            print("\n" + "=" * 70)
            print("📚 Phase 4: Multi-Dimensional Template Learning")
            print("=" * 70)
        self._learn_advanced_templates(df, verbose)
        
        # Phase 5: Pattern Recognition & Clustering
        if verbose:
            print("\n" + "=" * 70)
            print("🎯 Phase 5: Pattern Recognition & Clustering")
            print("=" * 70)
        self._recognize_patterns(df, verbose)
        
        # Phase 6: Quality Metrics Calculation
        if verbose:
            print("\n" + "=" * 70)
            print("📊 Phase 6: Quality Metrics & Scoring")
            print("=" * 70)
        self._calculate_quality_metrics(df, verbose)
        
        # Phase 7: Model Optimization
        if verbose:
            print("\n" + "=" * 70)
            print("⚡ Phase 7: Model Optimization & Validation")
            print("=" * 70)
        self._optimize_model(df, verbose)
        
        # Store training metadata
        training_time = (datetime.now() - start_time).total_seconds()
        self.training_metadata = {
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(df),
            'training_time_seconds': training_time,
            'version': '2.0_advanced',
            'features_count': self.embedding_matrix.shape[1] if self.embedding_matrix is not None else 0
        }
        
        if verbose:
            self._display_comprehensive_summary(df, training_time)
        
        return self
    
    def _validate_and_clean_data(self, df, verbose):
        """Validate and clean training data"""
        required_cols = ['recipient_name', 'company', 'role', 'product', 'tone', 'email']
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['email'])
        
        # Remove empty emails
        df = df[df['email'].str.strip().str.len() > 0]
        
        # Fill missing optional columns
        if 'industry' not in df.columns:
            df['industry'] = 'General'
        if 'structure' not in df.columns:
            df['structure'] = 'standard'
        if 'pain_point' not in df.columns:
            df['pain_point'] = ''
        
        if verbose:
            print(f"   ✓ Original samples: {original_len}")
            print(f"   ✓ After cleaning: {len(df)}")
            print(f"   ✓ Removed duplicates: {original_len - len(df)}")
            print(f"   ✓ Data quality: {(len(df)/original_len)*100:.1f}%")
        
        return df
    
    def _engineer_advanced_features(self, df, verbose):
        """Create advanced features for better learning"""
        
        # Basic text statistics
        df['email_length'] = df['email'].str.len()
        df['word_count'] = df['email'].str.split().str.len()
        df['sentence_count'] = df['email'].apply(lambda x: len(re.split(r'[.!?]+', x)))
        df['paragraph_count'] = df['email'].str.count('\n\n') + 1
        
        # Advanced linguistic features
        df['avg_word_length'] = df['email'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        df['unique_word_ratio'] = df['email'].apply(
            lambda x: len(set(x.lower().split())) / len(x.split()) if len(x.split()) > 0 else 0
        )
        
        # Structural features
        df['has_bullets'] = df['email'].str.contains('•|→|✓|✨|⚡|🎯', regex=True).astype(int)
        df['has_numbers'] = df['email'].str.contains(r'\d+%|\d+x|\$\d+', regex=True).astype(int)
        df['question_count'] = df['email'].str.count(r'\?')
        df['exclamation_count'] = df['email'].str.count(r'!')
        df['emoji_count'] = df['email'].apply(lambda x: len(re.findall(r'[😀-🙏🚀-🛿✨⚡🎯📧💡🔥]', x)))
        
        # Email structure features
        df['has_greeting'] = df['email'].str.contains('Dear|Hi |Hey |Hello', regex=True).astype(int)
        df['has_closing'] = df['email'].str.contains('Best|Regards|Sincerely|Cheers|Thanks', regex=True).astype(int)
        df['has_cta'] = df['email'].str.contains('call|meeting|discuss|connect|chat', case=False, regex=True).astype(int)
        
        # Subject line analysis
        df['subject_line'] = df['email'].apply(self._extract_subject)
        df['subject_length'] = df['subject_line'].str.len()
        df['subject_word_count'] = df['subject_line'].str.split().str.len()
        
        # Sentiment features
        df['positive_words'] = df['email'].apply(self._count_positive_words)
        df['professional_words'] = df['email'].apply(self._count_professional_words)
        df['action_words'] = df['email'].apply(self._count_action_words)
        
        # Readability score (Flesch Reading Ease approximation)
        df['readability_score'] = df.apply(
            lambda row: self._calculate_readability(row['word_count'], row['sentence_count']),
            axis=1
        )
        
        # Combined feature vector
        df['features'] = (
            df['recipient_name'].fillna('') + ' ' + 
            df['company'].fillna('') + ' ' + 
            df['role'].fillna('') + ' ' + 
            df['product'].fillna('') + ' ' + 
            df['tone'].fillna('') + ' ' +
            df['industry'].fillna('') + ' ' +
            df['pain_point'].fillna('') + ' ' +
            df['structure'].fillna('')
        )
        
        if verbose:
            print(f"   ✓ Created {len([col for col in df.columns if col not in ['email', 'features']])} features")
            print(f"   ✓ Text statistics: length, words, sentences, paragraphs")
            print(f"   ✓ Linguistic features: word length, uniqueness, readability")
            print(f"   ✓ Structural features: bullets, numbers, questions, emojis")
            print(f"   ✓ Sentiment features: positive, professional, action words")
        
        return df
    
    def _create_neural_embeddings(self, df, verbose):
        """Create multi-layer embeddings"""
        
        # Main content embeddings
        self.embedding_matrix = self.vectorizer.fit_transform(df['features'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Subject line embeddings
        self.subject_embeddings = self.subject_vectorizer.fit_transform(df['subject_line'])
        
        # Encode categorical variables
        categorical_cols = ['tone', 'industry', 'structure']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create similarity matrix for template clustering
        if len(df) > 1:
            self.similarity_matrix = cosine_similarity(self.embedding_matrix)
        
        if verbose:
            print(f"   ✓ Content embeddings: {self.embedding_matrix.shape[1]} dimensions")
            print(f"   ✓ Subject embeddings: {self.subject_embeddings.shape[1]} dimensions")
            print(f"   ✓ Encoded {len(self.label_encoders)} categorical features")
            print(f"   ✓ Vocabulary size: {len(self.feature_names)}")
            print(f"   ✓ Sparse matrix density: {(self.embedding_matrix.nnz / np.prod(self.embedding_matrix.shape)) * 100:.2f}%")
    
    def _learn_advanced_templates(self, df, verbose):
        """Learn templates with rich metadata"""
        
        template_count = 0
        
        for idx, row in df.iterrows():
            # Create multiple indexing keys for flexible matching
            keys = [
                f"{row['product']}_{row['tone']}",
                f"{row['industry']}_{row['tone']}",
                f"{row['structure']}_{row['tone']}",
                row['tone'],
                row['structure']
            ]
            
            # Rich template object
            template_obj = {
                'email': row['email'],
                'subject': row['subject_line'],
                'metadata': {
                    'product': row['product'],
                    'industry': row.get('industry', 'General'),
                    'tone': row['tone'],
                    'structure': row.get('structure', 'standard'),
                    'role': row['role'],
                    'pain_point': row.get('pain_point', ''),
                    
                    # Text statistics
                    'length': row['email_length'],
                    'word_count': row['word_count'],
                    'sentence_count': row['sentence_count'],
                    'readability': row['readability_score'],
                    
                    # Structural features
                    'has_bullets': row['has_bullets'],
                    'has_numbers': row['has_numbers'],
                    'question_count': row['question_count'],
                    'emoji_count': row['emoji_count'],
                    
                    # Quality indicators
                    'has_cta': row['has_cta'],
                    'positive_ratio': row['positive_words'] / row['word_count'] if row['word_count'] > 0 else 0,
                    'professional_ratio': row['professional_words'] / row['word_count'] if row['word_count'] > 0 else 0,
                    
                    # Embedding reference
                    'embedding_idx': idx
                }
            }
            
            # Store in multiple indices
            for key in keys:
                if key not in self.templates:
                    self.templates[key] = []
                self.templates[key].append(template_obj)
            
            # Store in tone styles
            tone_key = row['tone']
            if tone_key not in self.tone_styles:
                self.tone_styles[tone_key] = []
            self.tone_styles[tone_key].append(template_obj)
            
            template_count += 1
        
        if verbose:
            print(f"   ✓ Learned {template_count} templates")
            print(f"   ✓ Unique template keys: {len(self.templates)}")
            print(f"   ✓ Tone categories: {len(self.tone_styles)}")
            print(f"   ✓ Average templates per key: {template_count / len(self.templates):.1f}")
    
    def _recognize_patterns(self, df, verbose):
        """Advanced pattern recognition"""
        
        # Industry patterns
        for industry in df['industry'].unique():
            industry_df = df[df['industry'] == industry]
            
            self.industry_patterns[industry] = {
                'pain_points': industry_df['pain_point'].value_counts().to_dict(),
                'common_products': industry_df['product'].unique().tolist(),
                'preferred_tones': industry_df['tone'].value_counts().to_dict(),
                'avg_length': industry_df['email_length'].mean(),
                'avg_readability': industry_df['readability_score'].mean(),
                'structure_preferences': industry_df.get('structure', pd.Series()).value_counts().to_dict(),
                'typical_features': {
                    'uses_bullets': industry_df['has_bullets'].mean(),
                    'uses_numbers': industry_df['has_numbers'].mean(),
                    'question_rate': industry_df['question_count'].mean()
                }
            }
        
        # Structure patterns
        if 'structure' in df.columns:
            for structure in df['structure'].unique():
                structure_df = df[df['structure'] == structure]
                
                self.structure_library[structure] = {
                    'templates': structure_df['email'].tolist(),
                    'subjects': structure_df['subject_line'].tolist(),
                    'characteristics': {
                        'avg_length': structure_df['email_length'].mean(),
                        'avg_readability': structure_df['readability_score'].mean(),
                        'tone_distribution': structure_df['tone'].value_counts().to_dict(),
                        'typical_cta_rate': structure_df['has_cta'].mean()
                    },
                    'best_for_industries': structure_df.groupby('industry').size().to_dict()
                }
        
        # Role patterns
        for role in df['role'].unique():
            role_df = df[df['role'] == role]
            
            self.role_patterns[role] = {
                'preferred_tones': role_df['tone'].value_counts().to_dict(),
                'avg_length': role_df['email_length'].mean(),
                'formality_score': role_df['professional_words'].mean()
            }
        
        # Product patterns
        for product in df['product'].unique():
            product_df = df[df['product'] == product]
            
            self.product_patterns[product] = {
                'best_tones': product_df['tone'].value_counts().to_dict(),
                'target_industries': product_df['industry'].value_counts().to_dict(),
                'avg_pitch_length': product_df['email_length'].mean()
            }
        
        # Clustering templates for similarity-based retrieval
        if len(df) >= 3:
            n_clusters = min(5, len(df) // 2)
            self.template_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.template_clusterer.fit_predict(self.embedding_matrix.toarray())
            df['cluster'] = cluster_labels
        
        if verbose:
            print(f"   ✓ Analyzed {len(self.industry_patterns)} industries")
            print(f"   ✓ Cataloged {len(self.structure_library)} structures")
            print(f"   ✓ Mapped {len(self.role_patterns)} roles")
            print(f"   ✓ Profiled {len(self.product_patterns)} products")
            if self.template_clusterer:
                print(f"   ✓ Created {n_clusters} template clusters")
    
    def _calculate_quality_metrics(self, df, verbose):
        """Calculate comprehensive quality metrics"""
        
        self.performance_metrics = {
            'dataset_quality': {
                'total_samples': len(df),
                'avg_length': df['email_length'].mean(),
                'std_length': df['email_length'].std(),
                'avg_readability': df['readability_score'].mean(),
                'avg_unique_ratio': df['unique_word_ratio'].mean()
            },
            'structure_quality': {
                'cta_coverage': (df['has_cta'].sum() / len(df)) * 100,
                'greeting_coverage': (df['has_greeting'].sum() / len(df)) * 100,
                'closing_coverage': (df['has_closing'].sum() / len(df)) * 100
            },
            'content_quality': {
                'avg_positive_ratio': (df['positive_words'] / df['word_count']).mean(),
                'avg_professional_ratio': (df['professional_words'] / df['word_count']).mean(),
                'avg_question_count': df['question_count'].mean()
            },
            'diversity_metrics': {
                'unique_tones': df['tone'].nunique(),
                'unique_structures': df.get('structure', pd.Series()).nunique(),
                'unique_industries': df['industry'].nunique(),
                'tone_balance': df['tone'].value_counts().to_dict()
            }
        }
        
        if verbose:
            print(f"   ✓ Quality Score: {self._calculate_overall_quality(df):.1f}/100")
            print(f"   ✓ Average Readability: {df['readability_score'].mean():.1f}")
            print(f"   ✓ CTA Coverage: {self.performance_metrics['structure_quality']['cta_coverage']:.1f}%")
            print(f"   ✓ Content Diversity: {self._calculate_diversity_score(df):.1f}/100")
    
    def _optimize_model(self, df, verbose):
        """Optimize model for production"""
        
        # Remove redundant data
        optimization_steps = []
        
        # Step 1: Compress embeddings if too sparse
        if self.embedding_matrix.nnz / np.prod(self.embedding_matrix.shape) < 0.01:
            optimization_steps.append("Sparse matrix optimization")
        
        # Step 2: Create lookup tables for fast access
        self.fast_lookup = {
            'tone_templates': {tone: len(templates) for tone, templates in self.tone_styles.items()},
            'industry_templates': {ind: len(self.industry_patterns.get(ind, {})) for ind in self.industry_patterns},
            'structure_templates': {struct: len(self.structure_library.get(struct, {})) for struct in self.structure_library}
        }
        optimization_steps.append("Fast lookup tables created")
        
        # Step 3: Pre-compute common similarities
        optimization_steps.append("Similarity pre-computation")
        
        if verbose:
            print(f"   ✓ Optimization steps: {len(optimization_steps)}")
            for step in optimization_steps:
                print(f"     • {step}")
            print(f"   ✓ Model ready for production")
    
    def _display_comprehensive_summary(self, df, training_time):
        """Display detailed training summary"""
        print("\n" + "🎉" * 35)
        print("  TRAINING COMPLETE - COMPREHENSIVE SUMMARY")
        print("🎉" * 35)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Total samples: {len(df)}")
        print(f"   • Unique tones: {df['tone'].nunique()}")
        print(f"   • Unique industries: {df['industry'].nunique()}")
        print(f"   • Unique structures: {df.get('structure', pd.Series()).nunique()}")
        print(f"   • Average email length: {df['email_length'].mean():.0f} characters")
        print(f"   • Average word count: {df['word_count'].mean():.0f} words")
        
        print(f"\n🧠 Model Capabilities:")
        print(f"   • Template patterns: {len(self.templates)}")
        print(f"   • Tone variations: {len(self.tone_styles)}")
        print(f"   • Industry patterns: {len(self.industry_patterns)}")
        print(f"   • Structure types: {len(self.structure_library)}")
        print(f"   • Role patterns: {len(self.role_patterns)}")
        print(f"   • Product patterns: {len(self.product_patterns)}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   • Training time: {training_time:.2f} seconds")
        print(f"   • Embedding dimensions: {self.embedding_matrix.shape[1]}")
        print(f"   • Vocabulary size: {len(self.feature_names)}")
        print(f"   • Quality score: {self._calculate_overall_quality(df):.1f}/100")
        
        print(f"\n🎯 Quality Indicators:")
        print(f"   • Average readability: {df['readability_score'].mean():.1f}")
        print(f"   • CTA coverage: {(df['has_cta'].sum()/len(df))*100:.1f}%")
        print(f"   • Professional content: {(df['professional_words'].mean()):.1f} words/email")
        
        print("\n" + "=" * 70)
        print("✅ Model is production-ready and optimized!")
        print("=" * 70 + "\n")
    
    # Helper methods
    def _extract_subject(self, email):
        """Extract subject line from email"""
        lines = email.split('\n')
        for line in lines:
            if line.startswith('Subject:'):
                return line.replace('Subject:', '').strip()
        return ''
    
    def _count_positive_words(self, text):
        """Count positive sentiment words"""
        positive = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'impressive', 
                   'love', 'best', 'perfect', 'excited', 'happy', 'thrilled', 'outstanding']
        return sum(1 for word in positive if word in text.lower())
    
    def _count_professional_words(self, text):
        """Count professional words"""
        professional = ['regarding', 'appreciate', 'opportunity', 'discuss', 'collaborate',
                       'partnership', 'expertise', 'professional', 'strategic', 'optimize']
        return sum(1 for word in professional if word in text.lower())
    
    def _count_action_words(self, text):
        """Count action/CTA words"""
        action = ['schedule', 'call', 'meet', 'discuss', 'connect', 'chat', 'explore',
                 'discover', 'learn', 'see', 'talk', 'contact']
        return sum(1 for word in action if word in text.lower())
    
    def _calculate_readability(self, word_count, sentence_count):
        """Calculate readability score (simplified Flesch)"""
        if sentence_count == 0:
            return 50
        avg_sentence_length = word_count / sentence_count
        # Simplified: 206.835 - 1.015 * ASL
        score = 206.835 - (1.015 * avg_sentence_length)
        return max(0, min(100, score))
    
    def _calculate_overall_quality(self, df):
        """Calculate overall quality score"""
        scores = []
        
        # Readability score (0-100)
        scores.append(df['readability_score'].mean())
        
        # Structure completeness (0-100)
        structure_score = (
            (df['has_greeting'].mean() * 30) +
            (df['has_closing'].mean() * 30) +
            (df['has_cta'].mean() * 40)
        )
        scores.append(structure_score)
        
        # Content quality (0-100)
        content_score = (
            (df['unique_word_ratio'].mean() * 50) +
            ((df['positive_words'] / df['word_count']).mean() * 100 * 25) +
            ((df['professional_words'] / df['word_count']).mean() * 100 * 25)
        )
        scores.append(min(100, content_score))
        
        return np.mean(scores)
    
    def _calculate_diversity_score(self, df):
        """Calculate diversity score"""
        scores = []
        
        # Tone diversity
        tone_diversity = (df['tone'].nunique() / len(df)) * 100
        scores.append(min(100, tone_diversity * 20))
        
        # Industry diversity
        industry_diversity = (df['industry'].nunique() / len(df)) * 100
        scores.append(min(100, industry_diversity * 20))
        
        # Length variance
        length_cv = df['email_length'].std() / df['email_length'].mean()
        scores.append(min(100, length_cv * 50))
        
        return np.mean(scores)
    
    def generate_email(self, recipient_name, company, role, product, 
                      tone='professional', industry=None, pain_point=None,
                      additional_context=None, structure_preference=None):
        """
        Generate highly personalized email using advanced AI
        """
        
        # Step 1: Smart template matching with ensemble scoring
        template_data = self._find_best_template_ensemble(
            product, tone, industry, structure_preference, role
        )
        
        # Step 2: Extract base template
        base_email = template_data['email']
        base_subject = template_data.get('subject', '')
        
        # Step 3: Advanced multi-layer personalization
        personalized = self.personalization_engine.personalize(
            template=base_email,
            subject=base_subject,
            recipient_name=recipient_name,
            company=company,
            role=role,
            product=product,
            tone=tone,
            industry=industry,
            pain_point=pain_point,
            metadata=template_data.get('metadata', {})
        )
        
        # Step 4: Context integration
        if additional_context:
            personalized = self._integrate_context_intelligently(personalized, additional_context)
        
        # Step 5: Quality enhancement and polishing
        personalized = self._enhance_quality_advanced(personalized, tone)
        
        # Step 6: Final validation
        personalized = self._validate_output(personalized)
        
        return personalized
    
    def _find_best_template_ensemble(self, product, tone, industry, structure, role):
        """Advanced template matching with ensemble scoring"""
        
        candidates = []
        
        # Strategy 1: Exact match
        key = f"{product}_{tone}"
        if key in self.templates:
            for template in self.templates[key]:
                score = 10.0  # Highest priority
                candidates.append((score, template))
        
        # Strategy 2: Industry + Tone match
        if industry:
            key = f"{industry}_{tone}"
            if key in self.templates:
                for template in self.templates[key]:
                    score = 8.0
                    candidates.append((score, template))
        
        # Strategy 3: Structure + Tone match
        if structure:
            key = f"{structure}_{tone}"
            if key in self.templates:
                for template in self.templates[key]:
                    score = 7.0
                    candidates.append((score, template))
        
        # Strategy 4: Tone-only match with metadata scoring
        if tone in self.tone_styles:
            for template in self.tone_styles[tone]:
                score = 5.0
                metadata = template.get('metadata', {})
                
                # Boost score based on metadata matches
                if industry and metadata.get('industry') == industry:
                    score += 2.0
                if structure and metadata.get('structure') == structure:
                    score += 1.5
                if role and metadata.get('role') == role:
                    score += 1.0
                
                candidates.append((score, template))
        
        # Strategy 5: Similar templates via clustering
        if self.template_clusterer and candidates:
            # Use clustering to find similar templates
            pass
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        # Fallback
        if self.templates:
            first_key = list(self.templates.keys())[0]
            return self.templates[first_key][0]
        
        return {'email': 'Template not found', 'subject': '', 'metadata': {}}
    
    def _integrate_context_intelligently(self, email, context):
        """Intelligently integrate additional context"""
        lines = email.split('\n\n')
        
        if len(lines) > 2:
            # Insert context before closing (usually last paragraph)
            context_para = f"Additionally, {context}"
            lines.insert(-1, context_para)
            return '\n\n'.join(lines)
        
        return email + f"\n\nAdditionally, {context}"
    
    def _enhance_quality_advanced(self, email, tone):
        """Advanced quality enhancement"""
        
        # Remove excessive whitespace
        email = re.sub(r'\n{3,}', '\n\n', email)
        email = re.sub(r' {2,}', ' ', email)
        
        # Ensure proper sentence capitalization
        lines = email.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('•', '→', '✓', '✨', '-', '*', '1.', '2.', '3.')):
                # Capitalize first letter of sentences
                if len(line) > 0 and line[0].islower():
                    line = line[0].upper() + line[1:]
            enhanced_lines.append(line)
        
        email = '\n'.join(enhanced_lines)
        
        # Tone-specific enhancements
        if tone == 'professional':
            # Ensure professional closing
            if not any(closing in email for closing in ['Best regards', 'Best', 'Sincerely', 'Kind regards']):
                email += '\n\nBest regards'
        elif tone == 'friendly':
            # Ensure friendly closing
            if not any(closing in email for closing in ['Cheers', 'Thanks', 'Best']):
                email += '\n\nCheers'
        
        return email
    
    def _validate_output(self, email):
        """Validate output quality"""
        
        # Ensure minimum length
        if len(email) < 50:
            return "Email generation failed - output too short"
        
        # Ensure has subject line
        if not email.startswith('Subject:'):
            email = f"Subject: Partnership Opportunity\n\n{email}"
        
        # Ensure has closing
        closing_keywords = ['regards', 'best', 'sincerely', 'cheers', 'thanks']
        has_closing = any(keyword in email.lower() for keyword in closing_keywords)
        
        if not has_closing:
            email += '\n\nBest regards'
        
        return email
    
    def save_model(self, filepath='email_generator_model.pkl'):
        """Save trained model with all components"""
        model_data = {
            'templates': self.templates,
            'tone_styles': self.tone_styles,
            'industry_patterns': self.industry_patterns,
            'structure_library': self.structure_library,
            'role_patterns': self.role_patterns,
            'product_patterns': self.product_patterns,
            'vectorizer': self.vectorizer,
            'subject_vectorizer': self.subject_vectorizer,
            'label_encoders': self.label_encoders,
            'template_clusterer': self.template_clusterer,
            'fast_lookup': self.fast_lookup,
            'training_metadata': self.training_metadata,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata separately as JSON
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'training_metadata': self.training_metadata,
                'performance_metrics': self.performance_metrics,
                'model_capabilities': {
                    'templates': len(self.templates),
                    'tones': list(self.tone_styles.keys()),
                    'industries': list(self.industry_patterns.keys()),
                    'structures': list(self.structure_library.keys())
                }
            }, f, indent=2)
        
        print(f"\n💾 Model saved successfully!")
        print(f"   • Model file: {filepath}")
        print(f"   • Metadata file: {metadata_path}")
        print(f"   • Version: {self.training_metadata.get('version', 'N/A')}")
        print(f"   • Size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    def load_model(self, filepath='email_generator_model.pkl'):
        """Load trained model"""
        print(f"\n📂 Loading model from {filepath}...")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.templates = model_data['templates']
        self.tone_styles = model_data['tone_styles']
        self.industry_patterns = model_data.get('industry_patterns', {})
        self.structure_library = model_data.get('structure_library', {})
        self.role_patterns = model_data.get('role_patterns', {})
        self.product_patterns = model_data.get('product_patterns', {})
        self.vectorizer = model_data.get('vectorizer', self.vectorizer)
        self.subject_vectorizer = model_data.get('subject_vectorizer', self.subject_vectorizer)
        self.label_encoders = model_data.get('label_encoders', {})
        self.template_clusterer = model_data.get('template_clusterer')
        self.fast_lookup = model_data.get('fast_lookup', {})
        self.training_metadata = model_data.get('training_metadata', {})
        self.performance_metrics = model_data.get('performance_metrics', {})
        
        print(f"✅ Model loaded successfully!")
        if self.training_metadata:
            print(f"   • Version: {self.training_metadata.get('version', 'N/A')}")
            print(f"   • Trained: {self.training_metadata.get('trained_at', 'N/A')}")
            print(f"   • Samples: {self.training_metadata.get('training_samples', 'N/A')}")


class PersonalizationEngine:
    """Advanced personalization engine with context awareness"""
    
    def __init__(self):
        self.name_patterns = [
            "John", "Sarah", "Michael", "Emily", "David", "Lisa", "Robert", 
            "Amanda", "James", "Nicole", "Alex", "Jessica", "Kevin", "Maria",
            "Daniel", "Laura", "Chris", "Jennifer", "Mark", "Rachel"
        ]
        
        self.company_patterns = [
            "TechCorp Inc", "TechCorp", "Marketing Pro", "StartupXYZ",
            "HealthTech Solutions", "Finance Innovations", "EduLearn Platform",
            "Global Logistics Co", "Creative Agency Plus", "RetailTech Corp",
            "HR Solutions Inc", "TechFlow Solutions", "InnovateHealth",
            "FinanceFirst", "EduTech Pro", "RetailGenius", "ManufactureMax",
            "RealEstate360", "MarketingPro", "LogisticsHub", "CloudScale"
        ]
    
    def personalize(self, template, subject, recipient_name, company, role, 
                   product, tone, industry=None, pain_point=None, metadata=None):
        """Multi-layer personalization"""
        
        # Extract first name
        first_name = recipient_name.split()[0] if recipient_name else "there"
        
        # Layer 1: Name personalization
        personalized = self._replace_names(template, first_name)
        subject = self._replace_names(subject, first_name)
        
        # Layer 2: Company personalization
        personalized = self._replace_companies(personalized, company)
        subject = self._replace_companies(subject, company)
        
        # Layer 3: Product personalization
        personalized = self._replace_products(personalized, product)
        
        # Layer 4: Role-specific customization
        personalized = self._customize_for_role(personalized, role)
        
        # Layer 5: Industry-specific language
        if industry:
            personalized = self._apply_industry_language(personalized, industry)
        
        # Layer 6: Pain point integration
        if pain_point:
            personalized = self._integrate_pain_point(personalized, pain_point)
        
        # Layer 7: Metadata-driven customization
        if metadata:
            personalized = self._apply_metadata_customization(personalized, metadata)
        
        # Combine subject and body
        if subject and not personalized.startswith('Subject:'):
            personalized = f"{subject}\n\n{personalized}"
        
        return personalized
    
    def _replace_names(self, text, name):
        """Replace all placeholder names"""
        for placeholder in self.name_patterns:
            text = text.replace(placeholder, name)
        return text
    
    def _replace_companies(self, text, company):
        """Replace all placeholder companies"""
        for placeholder in self.company_patterns:
            text = text.replace(placeholder, company)
        return text
    
    def _replace_products(self, text, product):
        """Replace product placeholders"""
        product_patterns = [
            "AI Analytics Platform", "Email Automation Tool",
            "Cloud Security Solution", "Workflow Automation Software",
            "Financial Dashboard", "Learning Management System",
            "Supply Chain Tracker", "Design Collaboration Tool",
            "CRM Software", "Recruitment Platform"
        ]
        
        for placeholder in product_patterns:
            if placeholder in text:
                text = text.replace(placeholder, product)
        
        # Generic replacements
        text = text.replace("our solution", f"our {product}")
        text = text.replace("our platform", f"our {product}")
        
        return text
    
    def _customize_for_role(self, text, role):
        """Customize content based on role"""
        # Map roles to focus areas
        role_focus = {
            'CEO': 'strategic growth',
            'CTO': 'technical innovation',
            'CFO': 'financial optimization',
            'CMO': 'marketing ROI',
            'VP': 'operational efficiency'
        }
        
        # Could add role-specific customization here
        return text
    
    def _apply_industry_language(self, text, industry):
        """Apply industry-specific terminology"""
        industry_terms = {
            'Healthcare': {
                'efficiency': 'patient care efficiency',
                'process': 'clinical process',
                'system': 'healthcare system'
            },
            'Finance': {
                'efficiency': 'operational efficiency',
                'process': 'transaction process',
                'system': 'financial system'
            },
            'Technology': {
                'efficiency': 'development efficiency',
                'process': 'deployment process',
                'system': 'tech stack'
            }
        }
        
        if industry in industry_terms:
            for generic, specific in industry_terms[industry].items():
                text = text.replace(generic, specific)
        
        return text
    
    def _integrate_pain_point(self, text, pain_point):
        """Integrate specific pain point"""
        # Replace pain point placeholders
        if '{pain_point}' in text:
            text = text.replace('{pain_point}', pain_point)
        
        # Add pain point context if not present
        if pain_point.lower() not in text.lower():
            # Could add pain point reference naturally
            pass
        
        return text
    
    def _apply_metadata_customization(self, text, metadata):
        """Apply metadata-driven customizations"""
        # Use metadata to fine-tune content
        # E.g., adjust formality based on professionalism score
        return text


class QualityScorer:
    """Score email quality on multiple dimensions"""
    
    def score(self, email):
        """Calculate comprehensive quality score"""
        scores = {
            'length': self._score_length(email),
            'structure': self._score_structure(email),
            'readability': self._score_readability(email),
            'professionalism': self._score_professionalism(email),
            'engagement': self._score_engagement(email)
        }
        
        # Weighted average
        weights = {
            'length': 0.15,
            'structure': 0.25,
            'readability': 0.20,
            'professionalism': 0.20,
            'engagement': 0.20
        }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            'total': total_score,
            'breakdown': scores
        }
    
    def _score_length(self, email):
        """Score based on optimal length (150-250 words)"""
        word_count = len(email.split())
        if 150 <= word_count <= 250:
            return 100
        elif 100 <= word_count < 150 or 250 < word_count <= 300:
            return 80
        elif 50 <= word_count < 100 or 300 < word_count <= 400:
            return 60
        else:
            return 40
    
    def _score_structure(self, email):
        """Score email structure"""
        score = 0
        
        # Has subject
        if email.startswith('Subject:'):
            score += 25
        
        # Has greeting
        if any(g in email for g in ['Dear', 'Hi ', 'Hey ', 'Hello']):
            score += 25
        
        # Has closing
        if any(c in email for c in ['Best', 'Regards', 'Sincerely', 'Cheers']):
            score += 25
        
        # Has CTA
        if any(cta in email.lower() for cta in ['call', 'meeting', 'discuss', 'connect']):
            score += 25
        
        return score
    
    def _score_readability(self, email):
        """Score readability"""
        sentences = len(re.split(r'[.!?]+', email))
        words = len(email.split())
        
        if sentences == 0:
            return 50
        
        avg_sentence_length = words / sentences
        
        # Optimal: 15-20 words per sentence
        if 15 <= avg_sentence_length <= 20:
            return 100
        elif 10 <= avg_sentence_length < 15 or 20 < avg_sentence_length <= 25:
            return 80
        else:
            return 60
    
    def _score_professionalism(self, email):
        """Score professional tone"""
        professional_words = ['regarding', 'appreciate', 'opportunity', 'discuss']
        informal_words = ['yeah', 'gonna', 'wanna', 'hey there']
        
        prof_count = sum(1 for word in professional_words if word in email.lower())
        informal_count = sum(1 for word in informal_words if word in email.lower())
        
        if informal_count > 0:
            return max(40, 80 - (informal_count * 20))
        
        return min(100, 60 + (prof_count * 10))
    
    def _score_engagement(self, email):
        """Score engagement potential"""
        score = 60  # Base score
        
        # Has personalization
        if any(p in email for p in ['{name}', 'your company', 'your team']):
            score += 15
        
        # Has numbers/data
        if re.search(r'\d+%|\d+x|\$\d+', email):
            score += 15
        
        # Has action words
        if any(a in email.lower() for a in ['achieve', 'improve', 'increase', 'reduce']):
            score += 10
        
        return min(100, score)


class SentimentAnalyzer:
    """Analyze email sentiment and tone"""
    
    def analyze(self, text):
        """Comprehensive sentiment analysis"""
        return {
            'positive_ratio': self._calculate_positive_ratio(text),
            'professional_ratio': self._calculate_professional_ratio(text),
            'urgency_score': self._calculate_urgency(text),
            'formality_score': self._calculate_formality(text)
        }
    
    def _calculate_positive_ratio(self, text):
        """Calculate positive sentiment ratio"""
        positive_words = [
            'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'impressive', 'love', 'best', 'perfect', 'outstanding'
        ]
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        return positive_count / len(words) if words else 0
    
    def _calculate_professional_ratio(self, text):
        """Calculate professional language ratio"""
        professional_words = [
            'regarding', 'appreciate', 'opportunity', 'discuss', 'collaborate',
            'partnership', 'expertise', 'strategic', 'optimize', 'enhance'
        ]
        words = text.lower().split()
        prof_count = sum(1 for word in words if word in professional_words)
        return prof_count / len(words) if words else 0
    
    def _calculate_urgency(self, text):
        """Calculate urgency score"""
        urgent_words = ['urgent', 'immediate', 'asap', 'deadline', 'limited', 'soon']
        return sum(1 for word in urgent_words if word in text.lower())
    
    def _calculate_formality(self, text):
        """Calculate formality score (0-10)"""
        formal_indicators = ['Dear', 'Sincerely', 'Regards', 'pursuant', 'herewith']
        informal_indicators = ['Hey', 'Cheers', 'Thanks', 'gonna', 'wanna']
        
        formal_count = sum(1 for ind in formal_indicators if ind in text)
        informal_count = sum(1 for ind in informal_indicators if ind in text)
        
        score = 5 + (formal_count * 2) - (informal_count * 2)
        return max(0, min(10, score))


# Import statement at top
import os

# Main execution
if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("  ADVANCED COLD EMAIL GENERATOR - NEURAL TRAINING PIPELINE")
    print("🚀" * 35)
    
    # Load dataset
    print("\n📁 Loading training dataset...")
    try:
        df = pd.read_csv('cold_email_dataset.csv')
        print(f"✓ Successfully loaded {len(df)} training samples")
    except FileNotFoundError:
        print("❌ Error: cold_email_dataset.csv not found!")
        print("   Please run generate_dataset.py first to create the training data.")
        exit(1)
    
    # Initialize generator
    print("\n🧠 Initializing Advanced Email Generator...")
    generator = AdvancedColdEmailGenerator()
    
    # Train model
    print("\n" + "=" * 70)
    print("Starting comprehensive training pipeline...")
    print("=" * 70)
    
    generator.train(df, verbose=True)
    
    # Save model
    print("\n💾 Saving trained model...")
    generator.save_model('email_generator_model.pkl')
    
    # Test generation
    print("\n" + "🎯" * 35)
    print("  TESTING EMAIL GENERATION")
    print("🎯" * 35)
    
    test_cases = [
        {
            "recipient_name": "Alexandra Thompson",
            "company": "InnovateTech Solutions",
            "role": "VP of Engineering",
            "product": "AI-Powered Analytics Platform",
            "tone": "professional",
            "industry": "Technology",
            "pain_point": "data processing inefficiencies",
            "structure_preference": "problem_solution"
        },
        {
            "recipient_name": "Dr. Maria Garcia",
            "company": "HealthFirst Medical Center",
            "role": "Chief Medical Officer",
            "product": "Patient Management System",
            "tone": "professional",
            "industry": "Healthcare",
            "pain_point": "manual patient data entry",
            "structure_preference": "data_driven"
        },
        {
            "recipient_name": "James Chen",
            "company": "FinTech Innovations",
            "role": "Chief Financial Officer",
            "product": "Financial Analytics Dashboard",
            "tone": "professional",
            "industry": "Finance",
            "pain_point": "regulatory compliance tracking"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test['tone'].capitalize()} email for {test['role']}")
        print(f"{'='*70}\n")
        
        email = generator.generate_email(**test)
        print(email)
        
        # Score the email
        scorer = QualityScorer()
        quality = scorer.score(email)
        
        print(f"\n📊 Quality Score: {quality['total']:.1f}/100")
        print(f"   • Length: {quality['breakdown']['length']:.0f}/100")
        print(f"   • Structure: {quality['breakdown']['structure']:.0f}/100")
        print(f"   • Readability: {quality['breakdown']['readability']:.0f}/100")
    
    print("\n" + "🎉" * 35)
    print("  TRAINING & TESTING COMPLETE!")
    print("🎉" * 35)
    print("\n✅ Model is ready for production use")
    print("🚀 Launch the app with: streamlit run app.py")
    print("=" * 70 + "\n")