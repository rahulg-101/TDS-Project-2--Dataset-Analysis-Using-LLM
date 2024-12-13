import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import base64
import requests
import sys
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

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
            headers=headers, 
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error in API request: {e}")
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
        return info
    except Exception as e:
        print(f"Error in get_basic_info: {e}")
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
        # Taking sample of 10 values or count of values present in the series, whichever is the minimum
        samples = {col: df[col].dropna().sample(n=min(10, df[col].count())).tolist() for col in df.columns}         
        return samples
    except Exception as e:
        print(f"Error in get_sample_values: {e}")
        return {}

def outlier_detection(df):
    """
    Detect the lower bound, upper bound and count of outliers in numeric columns. 
    Not keeping all the outliers due to input token limit of our LLM.
    """
    try:
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
        return outliers
    except Exception as e:
        print(f"Error in outlier_detection: {e}")
        return {}



def correlation_heatmap(df):
    """Create correlation heatmap for numeric columns and saving it as correlation.png"""
    try:
        num_df = df.select_dtypes(include=[np.number])
        if len(num_df.columns) < 2:
            return None
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation.png')
        plt.close()
        
        return 'correlation.png'
    except Exception as e:
        print(f"Error in correlation_heatmap: {e}")
        return None
    

def plot_group(df, cols, group_num):
    """Create distribution plot for a group of columns using subplot
    
    Parameters:
    df (pandas.DataFrame): The dataset to analyze.
    cols (list): List of column names to create distribution plots for.
    group_num (int): The number to identify the group of plots.

    Returns:
    str: The path of the saved combined distribution plot. Returns None if an error occurs.
    """
    try:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cols):  
            plt.subplot(1, 3, i+1)          # Creating a subplot of histograms for 3 columns at a time
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = f'distributions_group_{group_num}.png'
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    except Exception as e:
        print(f"Error in plot_group: {e}")
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
    try:
        if not image_paths:
            return None

        # Calculating the number of rows required in the combined subplot
        no_plots = len(image_paths)
        rows = np.ceil(no_plots)
        
        # Establishing figure
        fig = plt.figure(figsize=(15, 20))
        
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
        
        plt.tight_layout()
        plt.savefig(output_path,bbox_inches='tight')  
        plt.close()
        
        # Delete all plots used to create the final subplot to free up memory
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                continue
        
        return output_path
    except Exception as e:
        print(f"Error in combine_plots: {e}")
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
        if num_rows > 3000 or len(numeric_cols) > 10:
            return False
        return True
    
    except Exception as e:
        print(f"Error in dynamic_analysis: {e}")
        return False


def distribution_plots(df):
    """
    Creates distribution plots for numeric columns, with behavior determined by dynamic_analysis.

    Parameters:
    df (pandas.DataFrame): The dataset to analyze.

    Returns:
    str: Path of the saved combined distribution plot or None if an error occurs.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None

        # Group numeric columns in batches of 3 for plotting
        col_groups = [numeric_cols[i:i + 3] for i in range(0, len(numeric_cols), 3)]
        
        # Check if full plotting is advised
        full_plotting = dynamic_analysis(df)
        plot_paths = []

        if full_plotting:
            # Create distribution plots for all columns
            for i, group_cols in enumerate(col_groups):
                plot_path = plot_group(df, group_cols, i)
                if plot_path:   
                    plot_paths.append(plot_path)
        else:
            # Just plot the first group if full plotting is not advised
            if col_groups:
                plot_path = plot_group(df, col_groups[0], 0)
                if plot_path:
                    plot_paths.append(plot_path)

        # Handle final output
        if len(plot_paths) == 1:
            os.rename(plot_paths[0], 'distributions.png')
            return 'distributions.png'
        
        final_path = combine_plots(plot_paths)
        
        return final_path

    except Exception as e:
        print(f"Error in distribution_plots: {e}")
        return None

def change_image_encoding(img):
    """Convert images to base64 string for LLM evaluation and returns them"""
    try:
        if not os.path.exists(img):
            return None
        with open(img, "rb") as image:
            return base64.b64encode(image.read()).decode('utf-8')
    except Exception as e:
        print(f"Error in change_image_encoding: {e}")
        return None

      

def initial_analyse(csv_file):
    """
    Perform initial analysis of the dataset using basic stats, sample values, and outliers detected (if any).
    This function reads a CSV file, extracts basic statistics, sample values, and outlier information,
    and then uses an LLM to generate a comprehensive analysis of the dataset.

    Parameters:
    csv_file (str): The path to the CSV file to be analyzed.

    Returns:
    str: A comprehensive analysis of the dataset including data overview, key patterns and relationships,
         notable outliers or anomalies, and potential insights and implications. If an error occurs during
         the analysis, it returns an error message.
    """
    try:
        df = pd.read_csv(csv_file, encoding='unicode_escape')

        basic_stats = get_basic_info(df)
        sample_values = get_sample_values(df)

        outliers = outlier_detection(df)

        # Initial Prompt
        analysis_prompt = {                                                     
            "role": "user",
            "content": f"""
    Given the following dataset statistics and sample values, analyze the dataset:
    Filename: {csv_file}
    Basic stats: {pd.json_normalize(basic_stats).to_dict(orient='records')[0]}
    Sample values: {json.dumps(sample_values, indent=2)}
    Outlier : {json.dumps(outliers, indent=2)}

    Please structure your analysis with clear, actionable sections:

    1. Data Overview
       - Dataset composition and size
       - Data quality assessment
       - Variable types and distributions

    2. Key Patterns & Relationships
       - Primary trends in the data
       - Notable correlations
       - Meaningful segments or clusters

    3. Anomalies & Special Cases
       - Statistical outliers with context
       - Unusual patterns or relationships
       - Data quality concerns

    4. Business Implications & Recommendations
       - Key insights for stakeholders
       - Specific action items
       - Areas for further investigation

    Focus on concrete, data-driven observations and avoid generic statements.
    Support each finding with specific numbers or examples from the data.
    Highlight practical implications rather than just statistical facts.
    """
        }
        analysis_response = llm_req([analysis_prompt])          # Calling LLM via requests for response

        if not analysis_response:
            return "Error in initial analysis LLM response"
        
        return analysis_response['choices'][0]['message']['content']

    except Exception as e:
        print(f"Error in initial_analyse: {e}")
        return "Error in analyzing data"




def generate_python_code(csv_file):
    """
    Generate python code based on basic stats, sample values and outliers detected(if any) in your dataset
    to perform various kind of analysis and create visualizations

    Parameters:
    csv_file (str): The path to the CSV file to be analyzed.

    Returns:
    str: Python code based on the analysis and visualization of the dataset. If an error occurs during the analysis, it returns an error message.
    """
    try:
        df = pd.read_csv(csv_file, encoding='unicode_escape')

        basic_stats = get_basic_info(df)
        sample_values = get_sample_values(df)
        outliers = outlier_detection(df)

        num_rows = basic_stats.get('Number of rows', 0)
        num_cols = basic_stats.get('Number of columns', 0)
        numeric_cols = basic_stats.get('numeric_columns', [])
        categorical_cols = basic_stats.get('categorical_columns', [])
        
        analysis_tasks = []
        if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            analysis_tasks.append("- Includes time series analysis")
        if len(numeric_cols) >= 2:
            analysis_tasks.append("- Performs clustering analysis")
        if outliers:
            analysis_tasks.append("- Handles outliers appropriately")
        if categorical_cols:
            analysis_tasks.append("- Conducts categorical analysis")
        analysis_tasks.append("- Regression analysis if there exists feature and label behavior in columns")


        # Prepare the prompt
        code_gen_prompt = {
            "role": "user",
            "content": f"""
            Given the following dataset information:
            Filename: {csv_file}
            Basic stats:  {pd.json_normalize(basic_stats).to_dict(orient='records')[0]}
            Sample values: {json.dumps(sample_values, indent=2)}
            Outliers: {json.dumps(outliers, indent=2)}

            Generate Python code to perform the following tasks:
            {json.dumps({"Tasks":analysis_tasks}, indent=2)}

            The code should:
            1. Be optimized for {num_rows} rows and {num_cols} columns
            2. Include appropriate visualizations and save them as *.PNG files
            3. Use efficient data processing methods
            4. Include proper error handling
            5. Be executable via the exec() method
            6. Be concise and optimized for lower-spec systems.
            7. While reading a csv file use : pd.read_csv(csv_file, encoding='unicode_escape') with same encoding
            8. Do not use inplace attribute for ANY pandas operations, instead use df[col] = df[col].method(value)
            9. Most Important, only perform either 1 or 2 analysis tasks that are listed above and not all of them, keep the usage simple and limited to libraries like pandas, numpy, seaborn, matplotlib and scikit-learn.

            Do not include any markdown or explanatory text, DO NOT EVEN ADD ```python at the beginning of your response.Provide pure Python code only.
            """
        }

        # Call LLM API
        code_gen_response = llm_req([code_gen_prompt])
        if not code_gen_response or 'choices' not in code_gen_response:
            return "Error in generating_python_code() method LLM request"
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
    print(response)

    try:
        exec_globals = {}
        exec(response, {}, exec_globals)            # Trying to execute code 
        print("Execution completed successfully.")
        generated_plots = exec_globals.get("generated_plots", [])
        return generated_plots

    except SyntaxError as se:
        print(f"Syntax error in the generated code: {se}")
    except Exception as e:
        print(f"Error executing LLM-generated code: {e}")
    
    return []

def visual_analysis(csv_file):
    """
    Perform visual analysis on the plots and graphs generated from the given CSV file.

    This function generates correlation heatmaps and distribution plots from the data,
    converts them to base64 strings, and passes them to a Language Model (LLM) for further analysis.

    Parameters:
    csv_file (str): The path to the CSV file to be analyzed.

    Returns:
    str: A string containing the LLM's analysis of the visualizations in Markdown format.
         If an error occurs during the analysis, it returns an error message.
    """
    try:
        df = pd.read_csv(csv_file, encoding='unicode_escape')

        correlation_plot = correlation_heatmap(df)
        distribution_plot = distribution_plots(df)

        vision_prompt = {
            "role": "user",
            "content": [
                {"type": "text", "text": """Analyze these visualizations focussing on the following:
                 1. Correlation Analysis
                - Identify strongest positive and negative correlations
                - Highlight unexpected relationships
                 
                2. Distribution Insights
                - Characterize the shape and spread of distributions
                - Note any bimodality or skewness
                - Identify variables requiring transformation
                 
                3. Practical Applications
                - Recommend specific modeling approaches
                - Suggest feature engineering ideas
                - Outline potential business applications
                 
                and provide concrete, specific observations rather than general statements in Markdown format:"""}
            ]
        }

        if correlation_plot:
            vision_prompt["content"].append({
                "type": "image_url",
                "image_url": {
                    "detail": "low",
                    "url": f"data:image/png;base64,{change_image_encoding(correlation_plot)}"
                }
            })

        if distribution_plot:
            vision_prompt["content"].append({
                "type": "image_url",
                "image_url": {
                    "detail": "low",
                    "url": f"data:image/png;base64,{change_image_encoding(distribution_plot)}"
                }
            })

        vision_response = llm_req([vision_prompt])  # Calling LLM via requests for response

        if vision_response is None:
            raise RuntimeError("Failed to get LLM response for vision analysis.")
        vision_analysis = vision_response['choices'][0]['message']['content']
        
        return vision_analysis

    except ValueError as e:
        print(f"Error: Failed to generate correlation heatmap or distribution plots. Reason: {e}")
        return "Error: Failed to generate correlation heatmap or distribution plots."

    except RuntimeError as e:
        print(f"Error: Failed to get LLM response. Reason: {e}")
        return "Error: Failed to get LLM response."
    except Exception as e:
        print(f"Error: An unexpected error occurred. Reason: {e}")
        return "Error: An unexpected error occurred."

    
def main(csv_file):
    """
    Main function where all the steps above are combined to generate a final narration of the story which utilises 
    Function Calling to generate the overview, main story, insights, recommendations based on the prior analysis methods performed !!
    """

    try:
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' does not exist")       # Check whether csv file exists or not
            return
        
        ## Created a function object that can be passed to LLM request for function calling
        function = [
            {
                "name": "final_narration",
                "description": "Provide information on initial analysis, vision or visual analysis(if present) and Generated Code for analysis for the current dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "initial_analysis": {
                            "type": "string",
                            "description": "Provide a comprehensive analysis which covers data overview,key patterns and relationships,outliers and anomalies,potential insights",
                        },
                        "vision_analysis": {
                            "type": "string",
                            "description": "Contains base64 string representation of annotated correlation heatmaps, distribution plots or both and maybe other related plots and graphs",
                        },
                        "generated_code_for_analysis": {
                            "type": "string",
                            "description": "Includes python code that can be executed for further analysis on this dataset such as clustering, regression analysis, time series analysis etc.",
                        },
                    },
                    "required": ["initial_analysis", "vision_analysis", "generated_code_for_analysis"],
                },
            }
        ]

        # First, get all the analyses
        initial_analysis_result = initial_analyse(csv_file)
        vision_analysis_result = visual_analysis(csv_file)
        generated_code_result = generate_python_code(csv_file)
        run_generated_code(generated_code_result)

        def final_narration(initial_analysis, vision_analysis, generated_code_for_analysis):
            analysis = {
                "initial_analysis": initial_analysis,
                "vision_analysis": vision_analysis,
                "generated_code_for_analysis": generated_code_for_analysis
            }
            return json.dumps(analysis)

        # Create a combined analysis
        combined_analysis = final_narration(
            initial_analysis=initial_analysis_result,
            vision_analysis=vision_analysis_result,
            generated_code_for_analysis=generated_code_result
        )
        current_directory = os.getcwd() 
        png_files = [] 
        main_viz_files = []
        # Iterate over all files in the current directory 
        for file in os.listdir(current_directory): 
            if file.endswith(".png") and file not in ['distributions.png','correlation.png']: 
                png_files.append(file)
            if file in ['distributions.png','correlation.png']:
                main_viz_files.append(file)

        # Prompt for final narration
        final_prompt = f"""Combine the following analyses into a cohesive narrative:

        Initial Analysis 
        Visual Analysis
        Generated Code for analysis

        Create a final README.md that:
        1. Narrate a compelling story about the data 
        2. Highlights key insights
        3. Provides actionable recommendations
        4. Properly add inferences about visualizations and leverage the visual analysis retrieved to reference the {main_viz_files} visualizations
        4. Don't forget to add {main_viz_files} using file_name.png into README.md 
        5. Also Add any extra generated visualizations from paths {png_files} using file_name.png in your README.md file
        6. Include generated code for analysis and how it is suitable 
        7. Detailed conclusion 

        Format in Markdown with clear sections and don't add ```markdown in the beginning of your output"""

        # Make the final LLM call with function calling !! 
        final_response = llm_req([
            {"role": "user", "content": final_prompt},
            {"role": "function", "name": "final_narration", "content": combined_analysis}])        # Calling LLM via requests for response

        if not final_response:
            print("Error: Failed to get final response")
            return

        final_narrative = final_response['choices'][0]['message']['content']

        # Write response received from LLM to a README.md file
        with open('README.md', 'w') as f:
            f.write(final_narrative)

        print("Analysis complete. README.md has been generated.")

    except Exception as e:
        print(f"Error in main function: {e}")
        return



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_file>")
        sys.exit(1)
    
    main(sys.argv[1])
    
    
    

