import logging
import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ExperimentLogger:
    """Comprehensive experiment logging and analysis recording system"""
    
    def __init__(self, experiment_name: str, base_dir: str = "./experiments/"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # Create directory structure
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / self.experiment_id
        self.logs_dir = self.exp_dir / "logs"
        self.models_dir = self.exp_dir / "models"
        self.results_dir = self.exp_dir / "results"
        self.plots_dir = self.exp_dir / "plots"
        self.data_dir = self.exp_dir / "data"
        
        # Create all directories
        for dir_path in [self.logs_dir, self.models_dir, self.results_dir, 
                        self.plots_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize metrics tracking
        self.metrics_history = []
        self.training_logs = []
        self.api_usage = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_cost_estimate': 0.0,
            'rate_limit_hits': 0
        }
        
        # Start time tracking
        self.start_time = time.time()
        
        self.logger.info(f"Experiment {self.experiment_id} started")
        
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logger
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.logs_dir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _format_value(self, value: Any) -> str:
        """Format value for logging with appropriate precision"""
        if isinstance(value, (int, np.integer)):
            return str(value)
        elif isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        elif isinstance(value, (tuple, list)):
            return str(value)
        elif isinstance(value, np.ndarray):
            return f"array{value.shape}"
        elif isinstance(value, dict):
            return str(value)
        else:
            return str(value)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        config_path = self.exp_dir / "config.json"
        
        # Convert config to JSON-serializable format
        serializable_config = self._make_serializable(config)
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2, default=str)
        
        self.logger.info("Configuration saved")
        self.logger.info(f"Config: {json.dumps(serializable_config, indent=2, default=str)}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return f"array{obj.shape}"
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        dataset_path = self.data_dir / "dataset_info.json"
        serializable_info = self._make_serializable(dataset_info)
        
        with open(dataset_path, 'w') as f:
            json.dump(serializable_info, f, indent=2, default=str)
        
        self.logger.info("Dataset information saved")
        self.logger.info(f"Dataset: {json.dumps(serializable_info, indent=2, default=str)}")
    
    def log_training_step(self, step: int, phase: str, metrics: Dict[str, Any], 
                         additional_info: Optional[Dict] = None):
        """Log training step with metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'phase': phase,
            'metrics': self._make_serializable(metrics),
            'additional_info': self._make_serializable(additional_info) if additional_info else {}
        }
        
        self.training_logs.append(log_entry)
        
        # Format metrics for logging with proper type handling
        if metrics:
            metrics_parts = []
            for k, v in metrics.items():
                formatted_value = self._format_value(v)
                metrics_parts.append(f"{k}: {formatted_value}")
            metrics_str = ", ".join(metrics_parts)
        else:
            metrics_str = "no metrics"
        
        self.logger.info(f"Step {step} ({phase}): {metrics_str}")
        
        # Save training logs incrementally
        training_log_path = self.logs_dir / "training_logs.json"
        with open(training_log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
    
    def log_api_call(self, success: bool, cost_estimate: float = 0.0, 
                    rate_limited: bool = False):
        """Log API usage statistics"""
        self.api_usage['total_calls'] += 1
        if success:
            self.api_usage['successful_calls'] += 1
        else:
            self.api_usage['failed_calls'] += 1
        
        self.api_usage['total_cost_estimate'] += cost_estimate
        
        if rate_limited:
            self.api_usage['rate_limit_hits'] += 1
        
        # Save API usage
        api_usage_path = self.logs_dir / "api_usage.json"
        with open(api_usage_path, 'w') as f:
            json.dump(self.api_usage, f, indent=2)
    
    def log_evaluation_results(self, results: Dict[str, float], 
                              phase: str = "test", 
                              additional_metrics: Optional[Dict] = None):
        """Log evaluation results"""
        evaluation_entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'results': self._make_serializable(results),
            'additional_metrics': self._make_serializable(additional_metrics) if additional_metrics else {}
        }
        
        self.metrics_history.append(evaluation_entry)
        
        # Format results for logging
        if results:
            results_parts = []
            for k, v in results.items():
                formatted_value = self._format_value(v)
                results_parts.append(f"{k}: {formatted_value}")
            results_str = ", ".join(results_parts)
        else:
            results_str = "no results"
        
        self.logger.info(f"Evaluation ({phase}): {results_str}")
        
        # Save results
        results_path = self.results_dir / f"evaluation_{phase}.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_entry, f, indent=2)
        
        # Save metrics history
        metrics_history_path = self.results_dir / "metrics_history.json"
        with open(metrics_history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_ablation_results(self, ablation_results: Dict[str, Dict[str, float]]):
        """Log ablation study results"""
        serializable_results = self._make_serializable(ablation_results)
        
        ablation_path = self.results_dir / "ablation_study.json"
        with open(ablation_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info("Ablation study results:")
        for variant, metrics in ablation_results.items():
            if metrics:
                metrics_parts = []
                for k, v in metrics.items():
                    formatted_value = self._format_value(v)
                    metrics_parts.append(f"{k}: {formatted_value}")
                metrics_str = ", ".join(metrics_parts)
            else:
                metrics_str = "no metrics"
            self.logger.info(f"  {variant}: {metrics_str}")
    
    def log_model_info(self, model_path: str, model_size: int, 
                      training_time: float, parameters: Dict[str, Any]):
        """Log model information"""
        model_info = {
            'model_path': model_path,
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'training_time_seconds': training_time,
            'training_time_formatted': f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s",
            'parameters': self._make_serializable(parameters),
            'timestamp': datetime.now().isoformat()
        }
        
        model_info_path = self.models_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Model saved: {model_path} ({model_info['model_size_mb']:.2f} MB)")
        self.logger.info(f"Training time: {model_info['training_time_formatted']}")
    
    def create_visualization(self, data: Dict[str, Any], plot_type: str = "metrics"):
        """Create and save visualizations"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            # Fallback if seaborn style not available
            plt.style.use('default')
        
        try:
            if plot_type == "metrics":
                self._plot_metrics_over_time(data)
            elif plot_type == "ablation":
                self._plot_ablation_results(data)
            elif plot_type == "api_usage":
                self._plot_api_usage()
            elif plot_type == "confusion_matrix":
                self._plot_confusion_matrix(data)
            elif plot_type == "distribution":
                self._plot_distributions(data)
        except Exception as e:
            self.logger.error(f"Error creating {plot_type} visualization: {e}")
    
    def _plot_metrics_over_time(self, metrics_data: Dict[str, List[float]]):
        """Plot metrics over time"""
        if not metrics_data:
            self.logger.warning("No metrics data provided for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)
        
        # Plot different metrics
        metric_names = list(metrics_data.keys())
        
        for i, metric in enumerate(metric_names[:4]):
            row, col = i // 2, i % 2
            if isinstance(metrics_data[metric], list) and len(metrics_data[metric]) > 0:
                axes[row, col].plot(metrics_data[metric])
                axes[row, col].set_title(f'{metric}')
                axes[row, col].set_xlabel('Step')
                axes[row, col].set_ylabel(metric)
                axes[row, col].grid(True)
            else:
                axes[row, col].text(0.5, 0.5, f'No data for {metric}', 
                                  transform=axes[row, col].transAxes, 
                                  ha='center', va='center')
                axes[row, col].set_title(f'{metric}')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metrics_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_results(self, ablation_data: Dict[str, Dict[str, float]]):
        """Plot ablation study results"""
        if not ablation_data:
            self.logger.warning("No ablation data provided for plotting")
            return
        
        # Extract metrics
        variants = list(ablation_data.keys())
        if not variants:
            return
            
        # Get all unique metrics across variants
        all_metrics = set()
        for variant_metrics in ablation_data.values():
            if isinstance(variant_metrics, dict):
                all_metrics.update(variant_metrics.keys())
        
        metrics = list(all_metrics)
        if not metrics:
            return
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = []
            variant_names = []
            
            for variant in variants:
                if (isinstance(ablation_data[variant], dict) and 
                    metric in ablation_data[variant]):
                    values.append(ablation_data[variant][metric])
                    variant_names.append(variant)
            
            if values:
                bars = axes[i].bar(variant_names, values)
                axes[i].set_title(f'{metric} - Ablation Study')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "ablation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_api_usage(self):
        """Plot API usage statistics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # API calls breakdown
        successful = self.api_usage.get('successful_calls', 0)
        failed = self.api_usage.get('failed_calls', 0)
        
        if successful + failed > 0:
            labels = ['Successful', 'Failed']
            values = [successful, failed]
            colors = ['green', 'red']
            
            axes[0].pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[0].set_title('API Calls Success Rate')
        else:
            axes[0].text(0.5, 0.5, 'No API calls recorded', 
                        transform=axes[0].transAxes, ha='center', va='center')
            axes[0].set_title('API Calls Success Rate')
        
        # Cost and rate limits
        metrics = ['Total Calls', 'Rate Limit Hits', 'Est. Cost ($)']
        values = [
            self.api_usage.get('total_calls', 0),
            self.api_usage.get('rate_limit_hits', 0),
            self.api_usage.get('total_cost_estimate', 0)
        ]
        
        bars = axes[1].bar(metrics, values)
        axes[1].set_title('API Usage Statistics')
        axes[1].set_ylabel('Count / Cost')
        
        # Add value labels
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "api_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distributions(self, data: Dict[str, np.ndarray]):
        """Plot data distributions"""
        if not data:
            self.logger.warning("No distribution data provided for plotting")
            return
        
        n_plots = len(data)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = axes.reshape(1, -1)
        elif rows > 1 and cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (name, values) in enumerate(data.items()):
            if rows == 1 and cols == 1:
                ax = axes
            elif rows == 1:
                ax = axes[i]
            elif cols == 1:
                ax = axes[i]
            else:
                row, col = i // cols, i % cols
                ax = axes[row, col]
            
            if isinstance(values, np.ndarray) and values.size > 0:
                ax.hist(values.flatten(), bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'Distribution of {name}')
                ax.set_xlabel(name)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {name}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'Distribution of {name}')
        
        # Hide empty subplots
        if rows > 1 or cols > 1:
            for i in range(n_plots, rows * cols):
                if rows == 1:
                    axes[i].axis('off')
                elif cols == 1:
                    axes[i].axis('off')
                else:
                    row, col = i // cols, i % cols
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        total_time = time.time() - self.start_time
        
        report = {
            'experiment_info': {
                'name': self.experiment_name,
                'id': self.experiment_id,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'total_duration_seconds': total_time,
                'total_duration_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s"
            },
            'api_usage': self.api_usage,
            'training_summary': {
                'total_training_steps': len(self.training_logs),
                'phases_completed': list(set([log['phase'] for log in self.training_logs]))
            },
            'evaluation_summary': {
                'evaluations_completed': len(self.metrics_history),
                'final_metrics': self.metrics_history[-1] if self.metrics_history else {}
            }
        }
        
        # Save report
        report_path = self.exp_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        self.logger.info(f"Experiment completed in {report['experiment_info']['total_duration_formatted']}")
        self.logger.info(f"Report generated: {report_path}")
        
        return report
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate markdown report"""
        markdown_path = self.exp_dir / "README.md"
        
        with open(markdown_path, 'w') as f:
            f.write(f"# Experiment Report: {self.experiment_name}\n\n")
            f.write(f"**Experiment ID:** {self.experiment_id}\n\n")
            f.write(f"**Duration:** {report['experiment_info']['total_duration_formatted']}\n\n")
            
            # API Usage
            f.write("## API Usage\n\n")
            f.write(f"- Total API Calls: {self.api_usage['total_calls']}\n")
            f.write(f"- Successful Calls: {self.api_usage['successful_calls']}\n")
            f.write(f"- Failed Calls: {self.api_usage['failed_calls']}\n")
            f.write(f"- Rate Limit Hits: {self.api_usage['rate_limit_hits']}\n")
            f.write(f"- Estimated Cost: ${self.api_usage['total_cost_estimate']:.2f}\n\n")
            
            # Training Summary
            f.write("## Training Summary\n\n")
            f.write(f"- Total Training Steps: {report['training_summary']['total_training_steps']}\n")
            f.write(f"- Phases: {', '.join(report['training_summary']['phases_completed'])}\n\n")
            
            # Final Results
            if self.metrics_history:
                f.write("## Final Results\n\n")
                final_results = self.metrics_history[-1].get('results', {})
                for metric, value in final_results.items():
                    formatted_value = self._format_value(value)
                    f.write(f"- {metric}: {formatted_value}\n")
                f.write("\n")
            
            # Files
            f.write("## Generated Files\n\n")
            f.write("- `config.json` - Experiment configuration\n")
            f.write("- `logs/experiment.log` - Detailed logs\n")
            f.write("- `results/` - Evaluation results\n")
            f.write("- `models/` - Saved models\n")
            f.write("- `plots/` - Generated visualizations\n")
            f.write("- `data/` - Dataset information\n")
    
    def save_raw_data(self, data: Dict[str, Any], filename: str):
        """Save raw experimental data"""
        try:
            serializable_data = self._make_serializable(data)
            
            if filename.endswith('.csv'):
                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(serializable_data)
                df.to_csv(self.data_dir / filename, index=False)
            elif filename.endswith('.npy'):
                # Save numpy array
                if isinstance(data, np.ndarray):
                    np.save(self.data_dir / filename, data)
                else:
                    self.logger.warning(f"Cannot save non-array data as .npy: {filename}")
                    return
            else:
                # Save as JSON
                with open(self.data_dir / filename, 'w') as f:
                    json.dump(serializable_data, f, indent=2, default=str)
            
            self.logger.info(f"Raw data saved: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving raw data {filename}: {e}")
    
    def export_to_csv(self):
        """Export all metrics to CSV files"""
        try:
            # Training logs to CSV
            if self.training_logs:
                training_df = pd.json_normalize(self.training_logs)
                training_df.to_csv(self.results_dir / "training_logs.csv", index=False)
            
            # Metrics history to CSV
            if self.metrics_history:
                metrics_df = pd.json_normalize(self.metrics_history)
                metrics_df.to_csv(self.results_dir / "metrics_history.csv", index=False)
            
            self.logger.info("Data exported to CSV files")
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")