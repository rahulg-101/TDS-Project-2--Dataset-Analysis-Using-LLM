# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "requests",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pybase64",
#   "tenacity",
#   "Pillow",
#   "python-dotenv",
#   "statsmodels",
#   "scikit-learn",
# ]
# ///



import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import base64
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
from dotenv import load_dotenv
import time
import warnings
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)


load_dotenv()

@retry(
    stop=stop_after_attempt(3),  # Try 3 times
    wait=wait_exponential(multiplier=1, min=4, max=10) ) # Wait between attempts
def llm_req(message, functions=None, stream=False):
    """
    Function to call LLM and retrieve response using the request method and AIPROXY TOKEN.

    Parameters:
    message (list): A list of dictionaries representing the messages to be sent to the LLM.
    functions (list, optional): A list of dictionaries representing the functions to be called by the LLM. Defaults to None.
    stream (bool, optional): A boolean indicating whether the response should be streamed. Defaults to False.

    Returns:
    dict: A dictionary representing the response from the LLM. If an error occurs during the request, returns None.
    """
    try:
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN')}"
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": message,
            "stream": stream
        }

        if functions:
            data["functions"] = functions

        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", 
            # "https://api.openai.com/v1/chat/completions",
            headers=headers, 
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in API request: {e}")
        return None



def get_basic_info(df):
    """
    Captures basic information and stats about your dataset
    
    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary containing basic information and stats about the dataset.
    """
    
    try:
        start = time.time()
        info = {
            'Number of rows': len(df),
            'Number of columns': len(df.columns),
            'column_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            info.update({
                'mean': {col: df[col].mean() for col in numeric_cols},
                'max': {col: df[col].max() for col in numeric_cols},
                'min': {col: df[col].min() for col in numeric_cols},
                'median': {col: df[col].median() for col in numeric_cols},
                '25th_percentile_Values': {col: df[col].quantile(0.25) for col in numeric_cols},
                '75th_percentile_Values': {col: df[col].quantile(0.75) for col in numeric_cols},
            })
        
        info.update({
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'unique_values': {col: df[col].nunique() for col in df.columns.tolist()},
        })
        end = time.time()
        print("for basic info",end - start)
        return info
    except Exception as e:
        logger.error(f"Error in get_basic_info: {e}")
        return {}

def get_sample_values(df):
    """
    This function returns a dictionary with sample values from each column in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary with sample values from each column in the DataFrame.
    """
    try:
        start = time.time()
        # Taking sample of 6 values or count of values present in the series, whichever is the minimum
        samples = {col: df[col].dropna().sample(n=min(6, df[col].count())).tolist() for col in df.columns}         
        end = time.time()
        print("For getting sample values",end - start)
        return samples
    except Exception as e:
        logger.error(f"Error in get_sample_values: {e}")
        return {}

def outlier_detection(df):
    """
    Detect the lower bound, upper bound and count of outliers in numeric columns. 
    Not keeping all the outliers due to input token limit of our LLM.

    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary with outlier values from each column in the DataFrame.
    """
    try:
        start = time.time()
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            lower_outliers = df.loc[df[col] < lower_bound, col]
            upper_outliers = df.loc[df[col] > upper_bound, col]

            # Creating a dictionary which contains sample of 5 or actual number that is present (whichever is minimum) of 
            # lower bound and upper bound outliers and their total count
            outliers[col] = {
                'lower_bound': lower_outliers.sample(n=min(5, len(lower_outliers))).tolist() if not lower_outliers.empty else [],
                'upper_bound': upper_outliers.sample(n=min(5, len(upper_outliers))).tolist() if not upper_outliers.empty else [],
                'count': len(lower_outliers) + len(upper_outliers)
            }
        end = time.time()
        print("For outlier detection",end - start)
        return outliers
    except Exception as e:
        logger.error(f"Error in outlier_detection: {e}")
        return {}

def return_none(series):
    """A helper function to return False if the series is empty"""

    if len(series) == 0:
        return False
    return True

def correlation_heatmap(df):
    """
    Create correlation heatmap for numeric columns and saving it as correlation.png

    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    path: The path to the correlation.png file    
    """
    try:
        start = time.time()
        num_df = df.select_dtypes(include=[np.number])
        if len(num_df.columns) < 2:
            return None
        
        plt.figure(figsize=(12,8))
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm',fmt= '.2f' )
        plt.title('Correlation Heatmap')
        plt.xlabel("Dataset Features")
        plt.ylabel("Dataset Features")
        plt.tight_layout()
        plt.savefig('correlation.png')
        plt.close()
        
        end = time.time()
        print("For correlation heatmap", end - start)
        return 'correlation.png'
    except Exception as e:
        logger.error(f"Error in correlation_heatmap: {e}")
        return None
    
def dynamic_analysis(df):
    """
    Analyzes the dataset to determine if full distribution plotting should be done
    based on dataset size and complexity.

    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    bool: True if full plotting should be done, False if limited plotting is advised.
    """
    try:
        num_rows = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if num_rows > 3000 or len(numeric_cols) > 6:
            return False
        return True
    
    except Exception as e:
        logger.error(f"Error in dynamic_analysis: {e}")
        return False    

def plot_group(df, cols, group_num):
    """Create distribution plot for a group of columns using subplot
    
    Parameters:
    df (pandas.DataFrame): The dataset to analyze.
    cols (list): List of column names to create distribution plots for.
    group_num (int): The number to identify the group of plots.

    Returns:
    str: The path of the saved combined distribution plot. Returns None if an error occurs.
    """
    start = time.time()
    try:
        fig = plt.figure(figsize=(12, 6))
        for i, col in enumerate(cols):  
            plt.subplot(1, 3, i+1)          # Creating a subplot of histograms for 3 columns at a time
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        output_path = f'distributions_group_{group_num}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    
        end = time.time()
        print("For plot group", end - start)
        return output_path
    except Exception as e:
        logger.error(f"Error in plot_group: {e}")
        return None

def combine_plots(image_paths, output_path='distributions.png'):
    """
    Combine multiple plots into a single figure
    Calculate number of rows for subplots
    If the number of plots is less than 3, create a single subplot instead of a grid

    Parameters:
    image_paths (list): List of paths to the images to be combined.
    output_path (str): The path where the combined image will be saved. Defaults to 'distributions.png'.

    Returns:
    str: The path of the saved combined image. Returns None if an error occurs.
    """
    start = time.time()
    
    try:
        if not image_paths:
            return None

        # Calculating the number of rows required in the combined subplot
        no_plots = len(image_paths)
        rows = np.ceil(no_plots)
        
        # Establishing figure
        fig = plt.figure(figsize=(12,rows*6))
        
        for i, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Combining all the subplots together to form a larger subplot 
            # capturing distributions of all the numberic columns
            ax = fig.add_subplot(rows, 1, i + 1)    
            ax.imshow(img_array)
            ax.axis('off')
        
        fig.suptitle("Distribution Analysis of Numerical Features", fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(output_path,bbox_inches='tight')  
        plt.close()
        
        # Delete all plots used to create the final subplot to free up memory
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                continue
        
        end = time.time()
        print("For combine plots", end - start)
        return output_path
    except Exception as e:
        logger.error(f"Error in combine_plots: {e}")
        return None




def distribution_plots(df):
    """
    Creates distribution plots for numeric columns, with behavior determined by dynamic_analysis.

    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    str: Path of the saved combined distribution plot or None if an error occurs.
    """
    try:
        start = time.time()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if return_none(numeric_cols)== False:
            return None

        # Group numeric columns in batches of 3 for plotting
        col_groups = [numeric_cols[i:i + 3] for i in range(0, len(numeric_cols), 3)]
        
        # Check if full plotting is advised
        full_plotting = dynamic_analysis(df)
        plot_paths = []

        if full_plotting:
            for i, group_cols in enumerate(col_groups):
                plot_path = plot_group(df, group_cols, i)
                if plot_path:   
                    plot_paths.append(plot_path)
        else:
            if col_groups:
                plot_path = plot_group(df, col_groups[0], 0)
                if plot_path:
                    plot_paths.append(plot_path)

                
        final_path = combine_plots(plot_paths)
        
        end = time.time()
        print("For distribution plots", end-start)
        return final_path

    except Exception as e:
        logger.error(f"Error in distribution_plots: {e}")
        return None

def change_image_encoding(img):
    """Convert images to base64 string for LLM evaluation and returns them
    
    Parameters:
    img (str): The path to the image saved

    Returns:
    str: String representation of the of images saved
    """
    try:
        start = time.time()
        if not os.path.exists(img):
            return None
        with open(img, "rb") as image:
            end = time.time()
            print("For change image encoding", end - start)
            return base64.b64encode(image.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in change_image_encoding: {e}")
        return None


def clustering(df,basic_info=None):
    """
    Perform K-means clustering on numeric data and find optimal number of clusters
    using silhouette score.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    basic_info (dict): Basic information about the dataset
    
    Returns:
    int: Optimal number of clusters
    """
    start = time.time()
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    stats = basic_info

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if return_none(numeric_cols) == False:
        return None

    # Create a copy of the numeric data
    numeric_data = df[numeric_cols].copy()

    # Handle missing values
    for col in numeric_cols:
        if col in stats['missing_values']:
            numeric_data[col] = numeric_data[col].fillna(numeric_data[col].median())
    
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_data)
    

    # K-means parameters
    kmeans_kwargs = {
        'init': 'k-means++',
        'n_init': 5,
        'max_iter': 20,
        'random_state': 42
    }

    # Calculate silhouette scores for different k values
    sil_coef_digits = []
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        sil_coef_digits.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(2, 15), sil_coef_digits, marker='o')
    plt.title('Silhouette Score Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.savefig('silhouette.png')
    plt.close()

    # Get optimal number of clusters (add 2 because range started from 2)
    n = sil_coef_digits.index(max(sil_coef_digits)) + 2
    
    end = time.time()
    print("K means cluserting",end-start)
    return n

def time_series_analysis(df):
    """
    Perform time series analysis on one numeric column against time-related columns.
    If the randomly selected numeric column is invalid (e.g., all-NaN or identical to the time column),
    another column is selected until a valid one is found or none remain.

    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    dict: A dictionary with hypothesis results for a column
    """
    def test_stationarity(p_value):
        if p_value <= 0.05:
            return "Strong evidence against the null hypothesis; the data is stationary."
        return "Weak evidence against the null hypothesis; the data is non-stationary."

    start = time.time()
    from statsmodels.tsa.stattools import adfuller

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if return_none(numeric_cols) == False:
        print("No numeric columns available for analysis.")
        return None

    # Identify potential time-related columns
    time_indicators = ['date', 'time', 'year', 'month', 'day', 'week']
    time_cols = [col for col in df.columns if any(indicator in col.lower() for indicator in time_indicators)]

    if not time_cols:
        raise ValueError("No time-related columns found in the dataset.")

    hypothesis_results = {}

    for time_col in time_cols:
        try:
            # Convert time column to datetime and drop invalid rows
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            time_series = df.dropna(subset=[time_col]).sort_values(time_col)

            # Randomly select a numeric column
            while numeric_cols:
                numeric_col = random.choice(numeric_cols)

                # Check if the column is valid
                if time_series[numeric_col].isnull().all():
                    print(f"Column {numeric_col} contains only NaN values. Selecting another column...")
                    numeric_cols.remove(numeric_col)
                    continue

                if time_series[numeric_col].equals(time_series[time_col]):
                    print(f"Column {numeric_col} is identical to {time_col}. Selecting another column...")
                    numeric_cols.remove(numeric_col)
                    continue

                # If a valid column is found, perform analysis
                series_data = time_series[numeric_col].ffill().bfill()

                # Create and save plot
                try:
                    plt.figure(figsize=(12, 6))
                    plt.plot(time_series[time_col], series_data)
                    plt.title(f'Time Series Analysis: {numeric_col} over {time_col}')
                    plt.xlabel(time_col)
                    plt.ylabel(numeric_col)
                    plt.grid(True)
                    plt.tight_layout()

                    safe_filename = f"timeseries_{time_col}_{numeric_col}".replace(" ", "_").replace("/", "_")
                    plt.savefig(f"{safe_filename}.png")
                    plt.close()

                except Exception as plot_error:
                    print(f"Error creating plot for {numeric_col} over {time_col}: {plot_error}")
                    return None

                # Perform ADF test
                try:
                    adf_result = adfuller(series_data.dropna())
                    adf_stat, p_value, _, _, critical_values, _ = adf_result

                    stationarity_result = test_stationarity(p_value)

                    hypothesis_results[f"{numeric_col}_{time_col}"] = {
                        "column": numeric_col,
                        "time_reference": time_col,
                        "adf_statistic": float(adf_stat),
                        "p_value": float(p_value),
                        "critical_values": critical_values,
                        "stationarity_test": stationarity_result,
                    }
                    break  # Exit the loop once a valid column is processed

                except Exception as adf_error:
                    print(f"Error performing ADF test for {numeric_col}: {adf_error}")
                    return None

            # Exit time_col loop once a valid numeric column is processed
            if hypothesis_results:
                break

        except Exception as e:
            logger.error(f"Error processing time column {time_col}: {str(e)}")
            continue

    end = time.time()
    print("Time series analysis completed in", end - start, "seconds.")
    return hypothesis_results

def dynamic_analysis_prompt(df: pd.DataFrame, basic_info: Dict, cluster,hypothesis) -> str:
    """
    Generate a dataset-specific analysis prompt based on characteristics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        basic_info (Dict): Basic dataset information
        cluster (int): A dictionary containing information about the number of clusters (if any) in the dataset.
        hypothesis (dict): A dictionary containing hypothesis test results (if present) for the dataset.
        
    Returns:
        str: Customized analysis prompt
    """
    num_rows = len(df)
    numeric_cols = basic_info.get('numeric_columns', [])
    categorical_cols = basic_info.get('categorical_columns', [])
    missing_values = basic_info.get('missing_values', {})
    clusters = cluster

    hypo = hypothesis
    # Build prompt sections based on data characteristics
    sections = []
    
    if numeric_cols:
        sections.append("""
        Numerical Analysis:
        - Identify key statistical patterns and relationships
        - Analyze distribution shapes and outliers
        - Examine correlations between numeric variables
        """)
        
    if categorical_cols:
        sections.append("""
        Categorical Analysis:
        - Analyze frequency distributions
        - Identify important category combinations
        - Examine relationships with numeric variables
        """)
        
    if any(missing_values.values()):
        sections.append("""
        Missing Data Analysis:
        - Assess impact of missing values
        - Identify patterns in missing data
        - Recommend handling strategies
        """)
        
    if len(df) > 1000:
        sections.append("""
        Large Dataset Considerations:
        - Focus on key trends and patterns
        - Identify significant subgroups
        - Recommend sampling strategies if needed
        """)
    
    if clusters:
        sections.append(f"""
        Cluster Analysis:
        - Identify clusters based on {cluster}
        - Analyze relationships between {clusters}
        - Explore cluster profiles and make recommendations
        """)
    
    if hypo:
        sections.append(f"""
        Hypothesis Testing:
        - Hypothesis Results: {hypo}
        - Determine if the hypothesis is supported by the data
        - Provide recommendations for further investigation
        """)
    
    # Combine sections into final prompt
    prompt = f"""
    Analyze this dataset with {num_rows} rows considering:
    
    {' '.join(sections)}
    
    Provide:
    1. Key insights and patterns
    2. Potential issues or limitations
    3. Specific recommendations for action for stakeholders
    4. Areas for further investigatio
    5. Refrences for findings
    """
    
    return prompt


def initial_analyse(csv_file,stats,sample,outlier,prompt):
    """
    Perform initial analysis of the dataset using basic stats, sample values, and outliers detected (if any).
    This function reads a CSV file, extracts basic statistics, sample values, and outlier information,
    and then uses an LLM to generate a comprehensive analysis of the dataset.

    Parameters:
    csv_file (str): The path to the CSV file to be analyzed.
    stats (dict): A dictionary containing basic statistics of the dataset.
    sample (dict): A dictionary containing sample values of the dataset.
    outlier (dict): A dictionary containing detected outliers in the dataset.
    

    Returns:
    str: A comprehensive analysis of the dataset including data overview, key patterns and relationships, notable outliers or anomalies, 
    and potential insights and implications. If an error occurs during the analysis, it returns an error message.
    """
    try:
        start = time.time()

        basic_stats = stats
        sample_values = sample

        outliers = outlier
        

        # Initial Prompt
        analysis_prompt = {
            "role": "user",
            "content": f"""
    Given the following dataset statistics and sample values, analyze the dataset:
    Filename: {csv_file}
    Basic stats: {pd.json_normalize(basic_stats).to_dict(orient='records')[0]}
    Sample values: {json.dumps(sample_values, indent=2)}
    Outlier : {json.dumps(outliers, indent=2)}
    

    {prompt}

    Focus on concrete, observations and avoid generic statements.
    Support each finding with specific numbers or examples from the data.
    Highlight practical implications rather than just statistical facts.
    """
        }

        analysis_response = llm_req([analysis_prompt])  # Calling LLM via requests for response
        if not analysis_response:
            return "Error in initial analysis LLM response"

        end = time.time()
        print("Initial analysis", end-start)
        # print(analysis_response)
        return analysis_response['choices'][0]['message']['content']

    except Exception as e:
        print(f"Error in initial_analyse: {e}")
        return "Error in analyzing data"





def generate_python_code(csv_file,stats):
    """
    Generate python code based on basic stats, sample values and outliers detected(if any) in your dataset
    to perform various kind of analysis and create visualizations

    Parameters:
    csv_file (str): The path to the CSV file to be analyzed.
    stats (dict): A dictionary containing basic statistics of the dataset.

    Returns:
    str: Python code based on the analysis of the dataset. If an error occurs during the analysis, it returns an error message.
    """
    try:
        start = time.time()

        # df = pd.read_csv(csv_file, encoding='unicode_escape')

        basic_stats = stats
        
        numeric_cols = basic_stats['numeric_columns']
                
        analysis_tasks = []
        if numeric_cols:
            analysis_tasks.append("- Performs Regression analysis by identifying features and target labels from numeric columns")
            analysis_tasks.append("- Feature Importance Analysis by identifying features and target labels from numeric columns")
        


        # Prepare the prompt
        code_gen_prompt = {
            "role": "user",
            "content": f"""
            Given the dataset details below:
            Filename: {csv_file}
            Numeric Columns : {numeric_cols}
            Generate Python code to perform **one task** from the following:
            {', '.join(analysis_tasks)}

            Code requirements:
            1. Perform the task and create **one supporting visualization** (saved as *.PNG).
            2. Use `df['col'] = df['col'].method()` instead of inplace methods.
            3. Restrict libraries to pandas, numpy,matplotlib, and sklearn.
            4. Handle missing values: fill numeric columns with medians, categorical with the most frequent value.
            5. Optimize for lower-spec systems: keep iterations or epochs low.
            6. Read the file with `encoding='unicode_escape'`.
            7. Work **only with numerical variables** for analysis and visualization.
            8. MOST IMPORTANT: Be directly executable via the exec() method
            
            Do not include any markdown or explanatory text, DO NOT EVEN ADD ```python at the beginning of your response.Provide pure Python code only.
            """
        }

        # Call LLM API
        code_gen_response = llm_req([code_gen_prompt])
        if not code_gen_response or 'choices' not in code_gen_response:
            return "Error in generating_python_code() method LLM request"
        
        end = time.time()
        print("Generate Python code",end-start)
        return code_gen_response['choices'][0]['message']['content']

    except Exception as e:
        print(f"Error in generate_python_code: {e}")
        return "Error in generating code"


def run_generated_code(response):
    """
    This function tries to run the python code generated by LLM via 'generate_python_code()' method

    Parameters:
    response (str): The python code generated by LLM via 'generate_python_code()' method
    
    Returns:
    list: A list of images (base64 strings) generated by the executed code. If an error occurs during execution, it returns an empty list.
    """
    # print(response)

    try:
        start = time.time()
        exec_globals = {}
        exec(response, {}, exec_globals)            # Trying to execute code 
        print("Execution completed successfully.")
        generated_plots = exec_globals.get("generated_plots", [])
        print("Run Generated code", time.time()-start)
        return generated_plots

    except SyntaxError as se:
        print(f"Syntax error in the generated code: {se}")
    except Exception as e:
        print(f"Error executing LLM-generated code: {e}")
    
    return []
    
@lru_cache(maxsize=32)
def analyze_plot(image_path: str, analysis_type: str) -> Optional[str]:
    """
    Analyze a single plot using LLM vision capabilities.
    
    Args:
        image_path (str): Path to the image file
        analysis_type (str): Type of analysis to perform
        
    Returns:
        Optional[str]: Analysis result or None if analysis fails
    """
    try:
        image_base64 = change_image_encoding(image_path)
        if not image_base64:
            logger.warning(f"Failed to encode {image_path}")
            return None

        vision_prompt = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Perform detailed {analysis_type} analysis on the following visualization.
                    Create a cohesive narrative that:
                    1. Flows naturally between different visualization types
                    2. Highlights relationships between findings
                    3. Emphasizes practical implications
                    4. Provides clear, actionable insights
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "detail": "low",
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }

        response = llm_req([vision_prompt])
        return response['choices'][0]['message']['content'] if response else None
    except Exception as e:
        logger.error(f"Error analyzing plot {image_path}: {e}")
        return None

def generate_plot_summaries() -> Dict[str, str]:
    """
    Generate summaries for all PNG plots in the current directory.
    
    Returns:
        Dict[str, str]: Dictionary mapping plot types to their analyses
    """
    try:
        start = time.time()
        vision_analysis_results = {}
        current_directory = os.getcwd()
        png_files = [f for f in os.listdir(current_directory) if f.endswith('.png')]
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for png_file in png_files:
                analysis_type = 'correlation' if 'correlation' in png_file else \
                              'distribution' if 'distribution' in png_file else \
                              'silhouette' if 'silhouette' in png_file else \
                              'timeseries' if 'timeseries' in png_file else 'custom'
                futures.append(executor.submit(analyze_plot, png_file, analysis_type))
            
            for f, png_file in zip(futures, png_files):
                result = f.result()
                if result:
                    vision_analysis_results[png_file] = result
        end = time.time()
        print("For Generate Plot summaries",end-start)
        return vision_analysis_results
    except Exception as e:
        logger.error(f"Error generating plot summaries: {e}")
        return {}

    
def dynamic_clustering_decision(df: pd.DataFrame, basic_info: Dict) -> bool:
    """
    Determine if clustering should be performed based on dataset characteristics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        basic_info (Dict): Basic dataset information
        
    Returns:
        bool: Whether clustering should be performed
    """
    num_rows = len(df)
    num_numeric_cols = len(basic_info.get('numeric_columns', []))
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # Define thresholds
    MAX_ROWS = 3000
    MAX_NUMERIC_COLS = 10
    MAX_MEMORY = 500  # MB
    
    return (num_rows < MAX_ROWS and 
            num_numeric_cols < MAX_NUMERIC_COLS and 
            memory_usage < MAX_MEMORY)


def process_dataset(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Optional[int], Optional[Dict]]:
    """
    Process dataset and generate various analyses.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Tuple containing basic stats, sample values, outliers, clusters, and hypothesis results
    """
    basic_stats = get_basic_info(df)
    sample_values = get_sample_values(df)
    outliers = outlier_detection(df)
    correlation_heatmap(df)
    distribution_plots(df)
    
    clusters = None
    if dynamic_clustering_decision(df, basic_stats):
        clusters = clustering(df, basic_stats)
        
    hypothesis = time_series_analysis(df)
    
    return basic_stats, sample_values, outliers, clusters, hypothesis


def main(csv_file: str) -> None:
    """
    Main function orchestrating the data analysis pipeline.
    
    Args:
        csv_file (str): Path to input CSV file
    """
    try:
        logger.info("Starting analysis pipeline")
        start_time = time.time()
        
        # Validate input file
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"File '{csv_file}' does not exist")
            
        # Read and process dataset
        df = pd.read_csv(csv_file, encoding='unicode_escape')
        basic_stats, sample_values, outliers, clusters, hypothesis = process_dataset(df)
        
        # Generate analyses
        analysis_prompt = dynamic_analysis_prompt(df, basic_stats,clusters, hypothesis)
        initial_analysis = initial_analyse(csv_file, basic_stats, sample_values, outliers, analysis_prompt)
        generated_code = generate_python_code(csv_file, basic_stats)
        
        # Execute generated code and analyze plots
        vision_analysis_results = generate_plot_summaries()
        run_generated_code(generated_code)
        current_directory = os.getcwd()
        png_files = [f for f in os.listdir(current_directory) if f.endswith('.png')]
        
        # Generate final narrative
        function_calling = [
            {
                "name": "final_narration",
                "description": "Integrate all analyses into a cohesive narrative",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "initial_analysis": {"type": "string"},
                        "vision_analysis": {"type": "string"},
                        "generated_code_for_analysis": {"type": "string"}
                    },
                    "required": ["initial_analysis", "vision_analysis", "generated_code_for_analysis"]
                }
            }
        ]
        
        final_prompt = f"""
        Create a comprehensive README.md in a narrative storytelling fashion that contains:
        
        1. Introduction to Dataset and Analysis Methods
           - Introduction of Dataset
           - Dataset Overview and Descriptions
           - Key characteristics and quality metrics
           
        2. Analysis & Findings
           - Statistical patterns and findings
           - Add plots present in {png_files} with same file names and respective interpretations.
           - Visual analysis results with clear references to plots
           - Advanced analytics results (clustering, time series, etc.)
           
        3. Data Insight & Technical Details
           - Insights reveleaded through different analysis methods
           - Generated code explanation and purpose
           - Methodology and approach
           
        4. Implications & Recommendations
           - Implications of working with this data
           - Business implications
           - Action items by stakeholder group
           - Areas for further investigation
           
        5. Conclusion
           - Summary of key findings
           - Next steps
        
        Do not add ```markdown at the beginning of markdown in readme.md.
        Ensure smooth transitions between sections.
        """
        
        final_response = llm_req(
            [
                {"role": "user", "content": final_prompt},
                {
                    "role": "function", 
                    "name": "final_narration", 
                    "content": json.dumps({
                        "initial_analysis": initial_analysis,
                        "generated_code_for_analysis": generated_code,
                        "vision_analysis": vision_analysis_results
                    })
                }
            ],
            functions=function_calling
        )
        
        if not final_response:
            raise RuntimeError("Failed to generate final response")
            
        # Write README
        with open('README.md', 'w') as f:
            f.write(final_response['choices'][0]['message']['content'])
            
        logger.info(f"Analysis complete in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_file>")
        sys.exit(1)
    
    main(sys.argv[1])
    
    
    

