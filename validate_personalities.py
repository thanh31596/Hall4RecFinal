#!/usr/bin/env python3
"""
Personality Data Validation Script
Validates and analyzes pre-generated personality JSON files
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
from collections import Counter

class PersonalityValidator:
    """Validates and analyzes personality JSON files"""
    
    def __init__(self, json_path: str = "./personalities.json"):
        self.json_path = Path(json_path)
        self.data = None
        self.personalities = None
        self.metadata = None
        
    def load_data(self) -> bool:
        """Load personality data from JSON"""
        if not self.json_path.exists():
            print(f"❌ File not found: {self.json_path}")
            return False
        
        try:
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
            
            self.personalities = self.data.get('personalities', {})
            self.metadata = self.data.get('metadata', {})
            
            print(f"✅ Loaded personality data: {len(self.personalities)} users")
            return True
            
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return False
    
    def validate_structure(self) -> Dict[str, Any]:
        """Validate JSON structure and required fields"""
        if not self.data:
            return {'valid': False, 'error': 'No data loaded'}
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check top-level structure
        required_top_level = ['personalities', 'metadata']
        for field in required_top_level:
            if field not in self.data:
                validation_results['errors'].append(f"Missing top-level field: {field}")
                validation_results['valid'] = False
        
        # Check metadata structure
        if self.metadata:
            expected_metadata = ['dataset', 'generation_timestamp', 'total_users', 'llm_model']
            for field in expected_metadata:
                if field not in self.metadata:
                    validation_results['warnings'].append(f"Missing metadata field: {field}")
        
        # Check personality entries
        if self.personalities:
            user_ids = list(self.personalities.keys())
            validation_results['stats']['total_entries'] = len(user_ids)
            
            # Check if user IDs are sequential
            try:
                numeric_ids = [int(uid) for uid in user_ids]
                expected_ids = list(range(max(numeric_ids) + 1))
                missing_ids = set(expected_ids) - set(numeric_ids)
                if missing_ids:
                    validation_results['warnings'].append(f"Missing user IDs: {sorted(missing_ids)}")
            except ValueError:
                validation_results['errors'].append("Non-numeric user IDs found")
                validation_results['valid'] = False
            
            # Check individual personality entries
            required_personality_fields = [
                'user_id', 'demographics', 'genre_preferences', 
                'personality_description', 'reflection', 'embedding', 
                'generation_status', 'timestamp'
            ]
            
            field_missing_count = Counter()
            invalid_embeddings = 0
            embedding_dimensions = []
            
            for uid, personality in self.personalities.items():
                for field in required_personality_fields:
                    if field not in personality:
                        field_missing_count[field] += 1
                
                # Check embedding
                embedding = personality.get('embedding', [])
                if not isinstance(embedding, list) or len(embedding) == 0:
                    invalid_embeddings += 1
                else:
                    embedding_dimensions.append(len(embedding))
            
            # Report field issues
            for field, count in field_missing_count.items():
                if count > 0:
                    validation_results['warnings'].append(
                        f"Field '{field}' missing in {count} entries"
                    )
            
            # Report embedding issues
            if invalid_embeddings > 0:
                validation_results['errors'].append(
                    f"{invalid_embeddings} entries have invalid embeddings"
                )
                validation_results['valid'] = False
            
            # Check embedding dimension consistency
            if embedding_dimensions:
                unique_dims = set(embedding_dimensions)
                if len(unique_dims) > 1:
                    validation_results['errors'].append(
                        f"Inconsistent embedding dimensions: {unique_dims}"
                    )
                    validation_results['valid'] = False
                else:
                    validation_results['stats']['embedding_dimension'] = list(unique_dims)[0]
        
        return validation_results
    
    def analyze_generation_quality(self) -> Dict[str, Any]:
        """Analyze the quality of generated personalities"""
        if not self.personalities:
            return {'error': 'No personality data available'}
        
        analysis = {
            'status_distribution': Counter(),
            'response_lengths': {
                'personality_description': [],
                'reflection': []
            },
            'demographic_coverage': {
                'age_distribution': [],
                'gender_distribution': Counter(),
                'occupation_distribution': Counter()
            },
            'genre_coverage': Counter(),
            'quality_issues': []
        }
        
        for uid, personality in self.personalities.items():
            # Status distribution
            status = personality.get('generation_status', 'unknown')
            analysis['status_distribution'][status] += 1
            
            # Response lengths
            desc = personality.get('personality_description', '')
            reflection = personality.get('reflection', '')
            
            analysis['response_lengths']['personality_description'].append(len(desc))
            analysis['response_lengths']['reflection'].append(len(reflection))
            
            # Check for quality issues
            if len(desc) < 50:
                analysis['quality_issues'].append(f"User {uid}: Short personality description")
            
            if len(reflection) < 30:
                analysis['quality_issues'].append(f"User {uid}: Short reflection")
            
            if 'unable to generate' in reflection.lower():
                analysis['quality_issues'].append(f"User {uid}: Failed reflection generation")
            
            # Demographics
            demographics = personality.get('demographics', {})
            if 'age' in demographics:
                analysis['demographic_coverage']['age_distribution'].append(demographics['age'])
            if 'gender' in demographics:
                analysis['demographic_coverage']['gender_distribution'][demographics['gender']] += 1
            if 'occupation' in demographics:
                analysis['demographic_coverage']['occupation_distribution'][demographics['occupation']] += 1
            
            # Genre preferences
            genre_prefs = personality.get('genre_preferences', {})
            for genre in genre_prefs.keys():
                analysis['genre_coverage'][genre] += 1
        
        return analysis
    
    def analyze_embeddings(self) -> Dict[str, Any]:
        """Analyze embedding properties"""
        if not self.personalities:
            return {'error': 'No personality data available'}
        
        embeddings = []
        valid_embeddings = 0
        
        for personality in self.personalities.values():
            embedding = personality.get('embedding', [])
            if isinstance(embedding, list) and len(embedding) > 0:
                try:
                    embedding_array = np.array(embedding, dtype=float)
                    embeddings.append(embedding_array)
                    valid_embeddings += 1
                except (ValueError, TypeError):
                    continue
        
        if not embeddings:
            return {'error': 'No valid embeddings found'}
        
        embeddings_matrix = np.array(embeddings)
        
        analysis = {
            'total_embeddings': len(embeddings),
            'embedding_dimension': embeddings_matrix.shape[1],
            'statistics': {
                'mean_norm': np.mean(np.linalg.norm(embeddings_matrix, axis=1)),
                'std_norm': np.std(np.linalg.norm(embeddings_matrix, axis=1)),
                'mean_values': np.mean(embeddings_matrix, axis=0),
                'std_values': np.std(embeddings_matrix, axis=0)
            },
            'diversity': {
                'pairwise_similarities': [],
                'mean_similarity': 0.0,
                'std_similarity': 0.0
            }
        }
        
        # Calculate pairwise similarities (sample if too large)
        n_samples = min(100, len(embeddings))
        if n_samples > 1:
            sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sample_embeddings = embeddings_matrix[sample_indices]
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(sample_embeddings)
            
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
            similarities = similarity_matrix[mask]
            
            analysis['diversity']['pairwise_similarities'] = similarities.tolist()
            analysis['diversity']['mean_similarity'] = np.mean(similarities)
            analysis['diversity']['std_similarity'] = np.std(similarities)
        
        return analysis
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive validation report"""
        if not self.load_data():
            return "Failed to load data"
        
        # Run all analyses
        structure_validation = self.validate_structure()
        quality_analysis = self.analyze_generation_quality()
        embedding_analysis = self.analyze_embeddings()
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PERSONALITY DATA VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"File: {self.json_path}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Metadata info
        if self.metadata:
            report_lines.append("METADATA:")
            for key, value in self.metadata.items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Structure validation
        report_lines.append("STRUCTURE VALIDATION:")
        report_lines.append(f"  Valid: {'✅' if structure_validation['valid'] else '❌'}")
        
        if structure_validation['errors']:
            report_lines.append("  Errors:")
            for error in structure_validation['errors']:
                report_lines.append(f"    - {error}")
        
        if structure_validation['warnings']:
            report_lines.append("  Warnings:")
            for warning in structure_validation['warnings']:
                report_lines.append(f"    - {warning}")
        
        if structure_validation['stats']:
            report_lines.append("  Statistics:")
            for key, value in structure_validation['stats'].items():
                report_lines.append(f"    {key}: {value}")
        report_lines.append("")
        
        # Generation quality
        if 'error' not in quality_analysis:
            report_lines.append("GENERATION QUALITY:")
            
            # Status distribution
            status_dist = quality_analysis['status_distribution']
            total_entries = sum(status_dist.values())
            report_lines.append("  Status Distribution:")
            for status, count in status_dist.items():
                percentage = (count / total_entries) * 100
                report_lines.append(f"    {status}: {count} ({percentage:.1f}%)")
            
            # Response lengths
            desc_lengths = quality_analysis['response_lengths']['personality_description']
            refl_lengths = quality_analysis['response_lengths']['reflection']
            
            if desc_lengths:
                report_lines.append("  Response Lengths:")
                report_lines.append(f"    Personality descriptions: {np.mean(desc_lengths):.0f} ± {np.std(desc_lengths):.0f} chars")
                report_lines.append(f"    Reflections: {np.mean(refl_lengths):.0f} ± {np.std(refl_lengths):.0f} chars")
            
            # Quality issues
            if quality_analysis['quality_issues']:
                report_lines.append("  Quality Issues:")
                for issue in quality_analysis['quality_issues'][:10]:  # Show first 10
                    report_lines.append(f"    - {issue}")
                if len(quality_analysis['quality_issues']) > 10:
                    report_lines.append(f"    ... and {len(quality_analysis['quality_issues']) - 10} more")
            
            report_lines.append("")
        
        # Embedding analysis
        if 'error' not in embedding_analysis:
            report_lines.append("EMBEDDING ANALYSIS:")
            report_lines.append(f"  Total embeddings: {embedding_analysis['total_embeddings']}")
            report_lines.append(f"  Dimension: {embedding_analysis['embedding_dimension']}")
            
            stats = embedding_analysis['statistics']
            report_lines.append(f"  Mean norm: {stats['mean_norm']:.3f} ± {stats['std_norm']:.3f}")
            
            diversity = embedding_analysis['diversity']
            if diversity['pairwise_similarities']:
                report_lines.append(f"  Mean similarity: {diversity['mean_similarity']:.3f} ± {diversity['std_similarity']:.3f}")
            
            report_lines.append("")
        
        # Overall assessment
        report_lines.append("OVERALL ASSESSMENT:")
        
        success_rate = 0
        if 'error' not in quality_analysis and quality_analysis['status_distribution']:
            successful = quality_analysis['status_distribution'].get('success', 0)
            total = sum(quality_analysis['status_distribution'].values())
            success_rate = (successful / total) * 100
        
        if structure_validation['valid'] and success_rate >= 80:
            assessment = "✅ EXCELLENT - Data is valid and high quality"
        elif structure_validation['valid'] and success_rate >= 60:
            assessment = "⚠️  GOOD - Data is valid but some quality concerns"
        elif structure_validation['valid']:
            assessment = "⚠️  POOR - Data is valid but low quality"
        else:
            assessment = "❌ INVALID - Data has structural issues"
        
        report_lines.append(f"  {assessment}")
        report_lines.append(f"  Success rate: {success_rate:.1f}%")
        
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        
        if success_rate < 80:
            report_lines.append("  - Consider regenerating personalities with better rate limiting")
        
        if structure_validation['errors']:
            report_lines.append("  - Fix structural issues before using in training")
        
        if 'error' not in quality_analysis and len(quality_analysis['quality_issues']) > 0:
            report_lines.append("  - Review quality issues and consider manual fixes")
        
        if structure_validation['valid'] and success_rate >= 80:
            report_lines.append("  - Data is ready for use in HallAgent4Rec training")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save report if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        
        return report_text
    
    def create_visualizations(self, output_dir: str = "./validation_plots/"):
        """Create visualization plots for personality data"""
        if not self.load_data():
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Analyze data
        quality_analysis = self.analyze_generation_quality()
        embedding_analysis = self.analyze_embeddings()
        
        # Set style
        plt.style.use('default')
        
        # 1. Status distribution pie chart
        if 'error' not in quality_analysis:
            status_dist = quality_analysis['status_distribution']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['green', 'orange', 'red', 'gray']
            ax.pie(status_dist.values(), labels=status_dist.keys(), autopct='%1.1f%%', colors=colors)
            ax.set_title('Generation Status Distribution')
            plt.tight_layout()
            plt.savefig(output_dir / 'status_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Response length distributions
            desc_lengths = quality_analysis['response_lengths']['personality_description']
            refl_lengths = quality_analysis['response_lengths']['reflection']
            
            if desc_lengths and refl_lengths:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.hist(desc_lengths, bins=30, alpha=0.7, color='blue')
                ax1.set_title('Personality Description Lengths')
                ax1.set_xlabel('Characters')
                ax1.set_ylabel('Frequency')
                
                ax2.hist(refl_lengths, bins=30, alpha=0.7, color='green')
                ax2.set_title('Reflection Lengths')
                ax2.set_xlabel('Characters')
                ax2.set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'response_lengths.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Demographics distribution
            demographics = quality_analysis['demographic_coverage']
            
            if demographics['age_distribution']:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Age distribution
                ax1.hist(demographics['age_distribution'], bins=20, alpha=0.7, color='purple')
                ax1.set_title('Age Distribution')
                ax1.set_xlabel('Age')
                ax1.set_ylabel('Frequency')
                
                # Gender distribution
                gender_dist = demographics['gender_distribution']
                ax2.bar(gender_dist.keys(), gender_dist.values(), alpha=0.7, color='orange')
                ax2.set_title('Gender Distribution')
                ax2.set_xlabel('Gender')
                ax2.set_ylabel('Count')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'demographics.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. Embedding analysis
        if 'error' not in embedding_analysis:
            embeddings = []
            for personality in self.personalities.values():
                embedding = personality.get('embedding', [])
                if isinstance(embedding, list) and len(embedding) > 0:
                    try:
                        embeddings.append(np.array(embedding, dtype=float))
                    except (ValueError, TypeError):
                        continue
            
            if embeddings:
                embeddings_matrix = np.array(embeddings)
                
                # Embedding norms
                norms = np.linalg.norm(embeddings_matrix, axis=1)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                ax1.hist(norms, bins=30, alpha=0.7, color='red')
                ax1.set_title('Embedding Norm Distribution')
                ax1.set_xlabel('L2 Norm')
                ax1.set_ylabel('Frequency')
                
                # PCA visualization (first 2 components)
                from sklearn.decomposition import PCA
                if embeddings_matrix.shape[0] > 2:
                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(embeddings_matrix)
                    
                    ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=20)
                    ax2.set_title('Embedding Space (PCA)')
                    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'embeddings.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Visualizations saved to: {output_dir}")

def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate HallAgent4Rec personality data')
    parser.add_argument('--file', default='./personalities.json', help='Path to personality JSON file')
    parser.add_argument('--report', help='Save report to file')
    parser.add_argument('--plots', help='Directory to save visualization plots')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    
    args = parser.parse_args()
    
    validator = PersonalityValidator(args.file)
    
    if args.summary:
        # Quick summary
        if validator.load_data():
            quality_analysis = validator.analyze_generation_quality()
            
            total_users = len(validator.personalities)
            successful = quality_analysis['status_distribution'].get('success', 0)
            success_rate = (successful / total_users) * 100 if total_users > 0 else 0
            
            print(f"\n📊 PERSONALITY DATA SUMMARY")
            print(f"   File: {args.file}")
            print(f"   Total users: {total_users}")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Status: {'✅ Ready' if success_rate >= 80 else '⚠️  Needs attention'}")
    else:
        # Full validation
        report = validator.generate_report(args.report)
        print(report)
        
        if args.plots:
            validator.create_visualizations(args.plots)

if __name__ == "__main__":
    main()
