import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig
import pandas as pd
from model.objectives.energyscore import EnergyScore
from model.objectives.crps import CRPS
from model.objectives.meanrmse import EnsembleMeanRMSE


def _generate_latex_table(
    all_metrics: Dict[str, Dict[str, float]],
    feature_name: str,
    lead_times: Dict[str, int],
    model_names: List[str]
) -> str:
    """
    Generate a LaTeX table for a specific feature with exact format matching the example.
    
    Args:
        all_metrics: Dictionary of all metrics for all models
        feature_name: Name of the feature to generate table for
        lead_times: Dictionary mapping time labels to lead times in hours
        model_names: List of model names (excluding 'deterministic')
    
    Returns:
        str: LaTeX table string
    """
    # Create a readable feature name for the caption
    display_feature_name = feature_name.replace('_', ' ').title()
    
    # Sort times in ascending order
    sorted_times = sorted(lead_times.items(), key=lambda x: x[1])
    time_labels = [t[0] for t in sorted_times]
    time_hours = [t[1] for t in sorted_times]
    
    # Start building the LaTeX table
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("    \\centering")
    latex.append("    \\setlength{\\tabcolsep}{3pt} % Reduce column spacing")
    latex.append("    \\scriptsize % Reduce font size for compactness")
    latex.append("    \\begin{tabular}{l S S S S | S S S S }")
    latex.append("        \\toprule")
    
    # Add the time headers - first two time points only
    first_time = f"\\multicolumn{{4}}{{c|}}{{\\textbf{{Lead Time: {time_hours[0]}h}}}}"
    second_time = f"\\multicolumn{{4}}{{c}}{{\\textbf{{Lead Time: {time_hours[1]}h}}}}"
    latex.append(f"        & {first_time} & {second_time} \\\\")
    
    # Add cmidrule separators
    latex.append("        \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
    
    # Add metric headers
    metrics_first = ["\\multicolumn{1}{c}{Energy}", "\\multicolumn{1}{c}{CRPS}", 
                   "\\multicolumn{1}{c}{RMSE}", "\\multicolumn{1}{c|}{Spread}"]
    metrics_second = ["\\multicolumn{1}{c}{Energy}", "\\multicolumn{1}{c}{CRPS}", 
                    "\\multicolumn{1}{c}{RMSE}", "\\multicolumn{1}{c}{Spread}"]
    latex.append(f"        & {' & '.join(metrics_first + metrics_second)} \\\\")
    
    # Find best values for bolding
    best_values = {}
    for time_label in time_labels:
        # For each metric at this time
        best_values[f"{time_label}_energy_score"] = float('inf')
        best_values[f"{time_label}_crps"] = float('inf')
        best_values[f"{time_label}_mean_rmse"] = float('inf')
        best_values[f"{time_label}_mean_spread"] = float('inf')
        
        # Find the best (minimum) values
        for model in model_names:
            metrics = all_metrics[model]
            for metric in ['energy_score', 'crps', 'mean_rmse', 'mean_spread']:
                key = f"{time_label}_{metric}_{feature_name}"
                if key in metrics and metrics[key] < best_values[f"{time_label}_{metric}"]:
                    best_values[f"{time_label}_{metric}"] = metrics[key]
    
    latex.append("        \\midrule")
    
    # Add deterministic row (only showing first two time points)
    det_row = ["Deterministic"]
    for time_label in time_labels[:2]:  # Only first two time points
        det_metrics = all_metrics['deterministic']
        
        # For Energy and CRPS, deterministic doesn't have values
        det_row.append("{-}")
        det_row.append("{-}")
        
        # For RMSE
        rmse_key = f"{time_label}_mean_rmse_{feature_name}"
        if rmse_key in det_metrics:
            det_row.append(f"\\num{{{det_metrics[rmse_key]:.2f}}}")
        else:
            det_row.append("{-}")
        
        # No spread for deterministic
        det_row.append("{-}")
    
    latex.append(f"        {' & '.join(det_row)} \\\\")
    
    # Add model rows (only first two time points)
    for model in model_names:
        model_row = [model]
        model_metrics = all_metrics[model]
        
        for time_label in time_labels[:2]:  # Only first two time points
            # Check if this is specific humidity or total precipitation that needs 5 decimal places
            decimal_places = 5 if ("specific_humidity" in feature_name or "total_precipitation" in feature_name) else 2
            
            # Energy Score
            energy_key = f"{time_label}_energy_score_{feature_name}"
            if energy_key in model_metrics:
                value = model_metrics[energy_key]
                if abs(value - best_values[f"{time_label}_energy_score"]) < 1e-6:
                    model_row.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row.append("{-}")
            
            # CRPS
            crps_key = f"{time_label}_crps_{feature_name}"
            if crps_key in model_metrics:
                value = model_metrics[crps_key]
                if abs(value - best_values[f"{time_label}_crps"]) < 1e-6:
                    model_row.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row.append("{-}")
            
            # Mean RMSE
            rmse_key = f"{time_label}_mean_rmse_{feature_name}"
            if rmse_key in model_metrics:
                value = model_metrics[rmse_key]
                if abs(value - best_values[f"{time_label}_mean_rmse"]) < 1e-6:
                    model_row.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row.append("{-}")
            
            # Mean-Spread
            spread_key = f"{time_label}_mean_spread_{feature_name}"
            if spread_key in model_metrics:
                value = model_metrics[spread_key]
                if abs(value - best_values[f"{time_label}_mean_spread"]) < 1e-6:
                    model_row.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row.append("{-}")
        
        latex.append(f"        {' & '.join(model_row)} \\\\")
    
    # Add midrule and headers for bottom half (last two time points)
    latex.append("        \\midrule")
    
    # Add the headers for time points 3 and 4
    third_time = f"\\multicolumn{{4}}{{c|}}{{\\textbf{{Lead Time: {time_hours[2]}h}}}}"
    fourth_time = f"\\multicolumn{{4}}{{c}}{{\\textbf{{Lead Time: {time_hours[3]}h}}}}"
    latex.append(f"        & {third_time} & {fourth_time} \\\\")
    
    # Add cmidrule separators for bottom section
    latex.append("        \\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
    
    # Add metric headers for bottom section (same as top)
    latex.append(f"        & {' & '.join(metrics_first + metrics_second)} \\\\")
    
    # Add midrule before data rows
    latex.append("        \\midrule")
    
    # Add deterministic row for bottom section (time points 3 and 4)
    det_row_bottom = ["Deterministic"]
    for time_label in time_labels[2:]:  # Only last two time points
        det_metrics = all_metrics['deterministic']
        
        # For Energy and CRPS, deterministic doesn't have values
        det_row_bottom.append("{-}")
        det_row_bottom.append("{-}")
        
        # For RMSE
        rmse_key = f"{time_label}_mean_rmse_{feature_name}"
        if rmse_key in det_metrics:
            det_row_bottom.append(f"\\num{{{det_metrics[rmse_key]:.2f}}}")
        else:
            det_row_bottom.append("{-}")
        
        # No spread for deterministic
        det_row_bottom.append("{-}")
    
    latex.append(f"        {' & '.join(det_row_bottom)} \\\\")
    
    # Add model rows for bottom section (time points 3 and 4)
    for model in model_names:
        model_row_bottom = [model]
        model_metrics = all_metrics[model]
        
        for time_label in time_labels[2:]:  # Only last two time points
            # Check if this is specific humidity or total precipitation that needs 5 decimal places
            decimal_places = 5 if ("specific_humidity" in feature_name or "total_precipitation" in feature_name) else 2
            
            # Energy Score
            energy_key = f"{time_label}_energy_score_{feature_name}"
            if energy_key in model_metrics:
                value = model_metrics[energy_key]
                if abs(value - best_values[f"{time_label}_energy_score"]) < 1e-6:
                    model_row_bottom.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row_bottom.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row_bottom.append("{-}")
            
            # CRPS
            crps_key = f"{time_label}_crps_{feature_name}"
            if crps_key in model_metrics:
                value = model_metrics[crps_key]
                if abs(value - best_values[f"{time_label}_crps"]) < 1e-6:
                    model_row_bottom.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row_bottom.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row_bottom.append("{-}")
            
            # Mean RMSE
            rmse_key = f"{time_label}_mean_rmse_{feature_name}"
            if rmse_key in model_metrics:
                value = model_metrics[rmse_key]
                if abs(value - best_values[f"{time_label}_mean_rmse"]) < 1e-6:
                    model_row_bottom.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row_bottom.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row_bottom.append("{-}")
            
            # Mean-Spread
            spread_key = f"{time_label}_mean_spread_{feature_name}"
            if spread_key in model_metrics:
                value = model_metrics[spread_key]
                if abs(value - best_values[f"{time_label}_mean_spread"]) < 1e-6:
                    model_row_bottom.append(f"\\textbf{{\\num{{{value:.{decimal_places}f}}}}}")
                else:
                    model_row_bottom.append(f"\\num{{{value:.{decimal_places}f}}}")
            else:
                model_row_bottom.append("{-}")
        
        latex.append(f"        {' & '.join(model_row_bottom)} \\\\")
    
    # Close the table
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append(f"    \\caption{{{display_feature_name} forecast metrics across different lead times.}}")
    latex.append(f"    \\label{{tab:{feature_name}_metrics}}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def run_comparison(
    run_data: Dict[str, Dict[str, torch.Tensor]],
    feature_dict: Dict[str, Any],
    feature_switch: Dict[str, bool],
    save_dir: str,
    cfg: DictConfig
) -> None:
    """
    Run comparison analysis between multiple model runs and generate a table of metrics.
    
    Args:
        run_data: Dictionary of model runs, with each run containing 'deterministic', 'diffusion', and 'truth' tensors
        feature_dict: Dictionary of feature information
        feature_switch: Dictionary indicating which features to analyze
        save_dir: Directory to save results
        cfg: Configuration object
    """
    print(f"Starting comparison of {len(run_data)} different runs")
    
    # Make sure the save directories exist
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Get a single ground truth to use for comparisons and a single deterministic result
    # (assuming deterministic models are identical across runs)
    ground_truth = next(iter(run_data.values()))['truth']
    deterministic_pred = next(iter(run_data.values()))['deterministic']
    
    # Calculate total timesteps
    total_timesteps = ground_truth.shape[0]
    
    # Define our evaluation points (T/4, T/2, 3T/4, T)
    eval_points = {
        'T/4': total_timesteps // 4,
        'T/2': total_timesteps // 2,
        'T3/4': 3 * total_timesteps // 4,
        'T': total_timesteps - 1  # -1 since indexing starts at 0
    }
    
    # Convert evaluation points to lead times in hours
    # Assuming 6 hours per timestep as per your config
    lead_times = {k: v * 6 for k, v in eval_points.items()}
    
    print(f"Evaluation points: {eval_points}")
    print(f"Lead times (hours): {lead_times}")
    
    # Initialize metrics
    probabilistic_metrics = {
        'energy_score': EnergyScore(),
        'crps': CRPS(),
        'mean_rmse': EnsembleMeanRMSE(),
    }
    
    # For deterministic metrics, use RMSE
    deterministic_metrics = {
        'rmse': lambda x, y: torch.sqrt(torch.mean((x - y)**2, dim=(-2, -1)))
    }
    
    # Initialize table data structure - for each feature we'll have a separate table
    feature_tables = {}
    
    # First, collect metrics for all models across all features
    all_metrics = {'deterministic': {}}
    
    # Process deterministic model
    det_metrics = _compute_deterministic_metrics(
        deterministic_pred, 
        ground_truth, 
        eval_points, 
        feature_dict, 
        feature_switch
    )
    all_metrics['deterministic'] = det_metrics
    
    # Process each diffusion model
    for run_name, data in run_data.items():
        diff_pred = data['diffusion']
        
        # Compute metrics for diffusion model
        diff_metrics = _compute_diffusion_metrics(
            diff_pred,
            ground_truth,
            eval_points,
            probabilistic_metrics,
            feature_dict,
            feature_switch
        )
        
        all_metrics[run_name] = diff_metrics
    
    # Create LaTeX tables for each feature
    for feature_name, feature_idx in feature_dict.items():
        if feature_idx is not None and feature_name in feature_switch and feature_switch[feature_name]:
            latex_table = _generate_latex_table(
                all_metrics, 
                feature_name, 
                lead_times, 
                run_data.keys()
            )
            
            # Save the LaTeX table to a file
            table_file = os.path.join(metrics_dir, f"{feature_name}_metrics_table.tex")
            with open(table_file, 'w') as f:
                f.write(latex_table)
            
            print(f"LaTeX table for {feature_name} saved to {table_file}")
    
    # Also save all metrics as CSV for easier processing
    # Flatten the nested structure for CSV output
    flattened_metrics = []
    for model_name, metrics in all_metrics.items():
        row = {'model': model_name}
        row.update(metrics)
        flattened_metrics.append(row)
    
    metrics_df = pd.DataFrame(flattened_metrics)
    
    # Save as CSV
    csv_path = os.path.join(metrics_dir, 'metrics_comparison.csv')
    metrics_df.to_csv(csv_path, index=False)
    
    print(f"Comparison completed successfully. Results saved to {metrics_dir}")

def _compute_deterministic_metrics(
    det_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    eval_points: Dict[str, int],
    feature_dict: Dict[str, Any],
    feature_switch: Dict[str, bool]
) -> Dict[str, float]:
    """Computes metrics for deterministic prediction at specific time points"""
    metrics = {}
    
    # RMSE calculation function
    rmse_fn = lambda x, y: torch.sqrt(torch.mean((x - y)**2, dim=(-2, -1)))
    
    for time_label, time_idx in eval_points.items():
        # Calculate metrics for each feature
        for feature_name, feature_idx in feature_dict.items():
            if feature_idx is not None and feature_name in feature_switch and feature_switch[feature_name]:
                # For deterministic model, we just need the RMSE at the specific timestep
                
                # Get prediction and ground truth at this timestep
                pred_t = det_pred[time_idx]
                truth_t = ground_truth[time_idx]  # Assuming ground truth has ensemble dim of 1
                
                # Extract feature data
                pred_feature = pred_t[feature_idx, :, :]
                truth_feature = truth_t[feature_idx, :, :]
                
                # Calculate RMSE
                rmse = rmse_fn(pred_feature, truth_feature).item()
                
                # Store metric values
                metrics[f"{time_label}_energy_score_{feature_name}"] = rmse  # Placeholder
                metrics[f"{time_label}_crps_{feature_name}"] = rmse  # Placeholder
                metrics[f"{time_label}_mean_rmse_{feature_name}"] = rmse
                metrics[f"{time_label}_mean_spread_{feature_name}"] = rmse  # For deterministic, use RMSE as placeholder
    
    return metrics

def _compute_diffusion_metrics(
    diff_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    eval_points: Dict[str, int],
    metrics: Dict[str, Any],
    feature_dict: Dict[str, Any],
    feature_switch: Dict[str, bool]
) -> Dict[str, float]:
    """Computes metrics for diffusion prediction at specific time points"""
    result_metrics = {}
    
    for time_label, time_idx in eval_points.items():
        # For each feature that is enabled
        for feature_name, feature_idx in feature_dict.items():
            if feature_idx is not None and feature_name in feature_switch and feature_switch[feature_name]:
                # Calculate metrics
                for metric_name, metric_fn in metrics.items():
                    # Get the full metrics computation (which returns spread, skill, score or just the score)
                    if metric_name in ['energy_score', 'crps']:
                        # These metrics return a tuple of (spread, skill, score)
                        metric_values = metric_fn.compute(diff_pred, ground_truth)
                        # Extract just the score at the specific timestep
                        score = metric_values[2][time_idx, feature_idx].item()
                    else:
                        # For mean_rmse, it returns a single value
                        metric_value = metric_fn.compute(diff_pred, ground_truth)
                        score = metric_value[time_idx, feature_idx].item()
                    
                    # Store in results
                    result_metrics[f"{time_label}_{metric_name}_{feature_name}"] = score
                
                # Calculate ||(mean-ground) - spread||_2 at the specific timestep
                # Extract the timestep data
                pred_t = diff_pred[time_idx]  # This includes all ensemble members
                truth_t = ground_truth[time_idx]  # Assuming ground truth has ensemble dim of 1
                
                # Extract feature data
                pred_feature = pred_t[:, feature_idx, :, :]
                truth_feature = truth_t[feature_idx, :, :]
                
                # Calculate mean prediction across ensemble
                mean_pred = pred_feature.mean(dim=0)
                
                # Calculate spread as standard deviation across ensemble
                spread = torch.std(pred_feature, dim=0)
                
                # Calculate ||(mean-ground) - spread||_2
                mean_ground_diff = torch.abs(mean_pred - truth_feature)
                mean_spread_diff = torch.norm(mean_ground_diff - spread, p=2).item()
                
                # Add the mean-spread metric
                result_metrics[f"{time_label}_mean_spread_{feature_name}"] = mean_spread_diff
    
    return result_metrics