import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

def load_experiment_data(experiment_path):
    """Load experiment data from directory"""
    exp_path = Path(experiment_path)
    
    data = {}
    
    # Load config
    config_path = exp_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data['config'] = json.load(f)
    
    # Load results
    results_dir = exp_path / "results"
    if results_dir.exists():
        for file in results_dir.glob("*.json"):
            with open(file) as f:
                data[file.stem] = json.load(f)
    
    # Load training logs
    logs_path = exp_path / "logs" / "training_logs.json"
    if logs_path.exists():
        with open(logs_path) as f:
            data['training_logs'] = json.load(f)
    
    # Load API usage
    api_path = exp_path / "logs" / "api_usage.json"
    if api_path.exists():
        with open(api_path) as f:
            data['api_usage'] = json.load(f)
    
    return data

def main():
    st.set_page_config(page_title="HallAgent4Rec Analysis Dashboard", layout="wide")
    
    st.title("🤖 HallAgent4Rec Analysis Dashboard")
    
    # Sidebar for experiment selection
    st.sidebar.header("Experiment Selection")
    
    experiments_dir = Path("./experiments")
    if experiments_dir.exists():
        experiment_folders = [d.name for d in experiments_dir.iterdir() if d.is_dir()]
        selected_experiment = st.sidebar.selectbox("Select Experiment", experiment_folders)
        
        if selected_experiment:
            exp_path = experiments_dir / selected_experiment
            data = load_experiment_data(exp_path)
            
            # Main dashboard
            col1, col2, col3 = st.columns(3)
            
            # Key metrics
            with col1:
                st.metric("Experiment", selected_experiment)
                if 'api_usage' in data:
                    st.metric("API Calls", data['api_usage'].get('total_calls', 0))
            
            with col2:
                if 'evaluation_main_evaluation' in data:
                    results = data['evaluation_main_evaluation']['results']
                    if 'HitRate@10' in results:
                        st.metric("Hit Rate@10", f"{results['HitRate@10']:.3f}")
            
            with col3:
                if 'api_usage' in data:
                    cost = data['api_usage'].get('total_cost_estimate', 0)
                    st.metric("Est. Cost", f"${cost:.2f}")
            
            # Training progress
            if 'training_logs' in data:
                st.header("Training Progress")
                
                training_df = pd.json_normalize(data['training_logs'])
                
                # Plot training metrics
                fig = px.line(training_df, x='step', y='metrics', 
                             title="Training Metrics Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # Evaluation results
            if 'evaluation_main_evaluation' in data:
                st.header("Evaluation Results")
                
                results = data['evaluation_main_evaluation']['results']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart of metrics
                    metrics_df = pd.DataFrame(list(results.items()), 
                                            columns=['Metric', 'Value'])
                    fig = px.bar(metrics_df, x='Metric', y='Value', 
                               title="Evaluation Metrics")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(metrics_df)
            
            # API Usage
            if 'api_usage' in data:
                st.header("API Usage Statistics")
                
                api_data = data['api_usage']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart of success/failure
                    success_data = {
                        'Status': ['Successful', 'Failed'],
                        'Count': [api_data.get('successful_calls', 0), 
                                api_data.get('failed_calls', 0)]
                    }
                    fig = px.pie(values=success_data['Count'], 
                               names=success_data['Status'],
                               title="API Call Success Rate")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("API Statistics")
                    st.write(f"**Total Calls:** {api_data.get('total_calls', 0)}")
                    st.write(f"**Successful:** {api_data.get('successful_calls', 0)}")
                    st.write(f"**Failed:** {api_data.get('failed_calls', 0)}")
                    st.write(f"**Rate Limit Hits:** {api_data.get('rate_limit_hits', 0)}")
                    st.write(f"**Estimated Cost:** ${api_data.get('total_cost_estimate', 0):.2f}")
            
            # Ablation study results
            if 'ablation_study' in data:
                st.header("Ablation Study Results")
                
                ablation_data = data['ablation_study']
                
                # Convert to DataFrame for plotting
                ablation_rows = []
                for variant, metrics in ablation_data.items():
                    for metric, value in metrics.items():
                        ablation_rows.append({
                            'Variant': variant,
                            'Metric': metric,
                            'Value': value
                        })
                
                ablation_df = pd.DataFrame(ablation_rows)
                
                fig = px.bar(ablation_df, x='Variant', y='Value', color='Metric',
                           title="Ablation Study Results",
                           barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            # Configuration
            if 'config' in data:
                st.header("Experiment Configuration")
                
                config = data['config']
                
                # Display config as formatted JSON
                st.json(config)
            
            # Raw data download
            st.header("Download Results")
            
            if st.button("Generate Summary Report"):
                summary = {
                    'experiment': selected_experiment,
                    'timestamp': data.get('config', {}).get('timestamp', ''),
                    'results': data.get('evaluation_main_evaluation', {}).get('results', {}),
                    'api_usage': data.get('api_usage', {}),
                    'config': data.get('config', {})
                }
                
                st.download_button(
                    label="Download Summary JSON",
                    data=json.dumps(summary, indent=2),
                    file_name=f"{selected_experiment}_summary.json",
                    mime="application/json"
                )
    else:
        st.warning("No experiments found. Run an analysis first!")

if __name__ == "__main__":
    main()