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
    """Function to call LLM and retrieve response using the request method and AIPROXY TOKEN"""
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
    """Captures basic information and stats about your dataset"""
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
    """Create distribution plot for a group of columns using subplot"""
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

def distribution_plots(df):
    """Create distribution plots for numeric columns"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None

        col_groups = [numeric_cols[i:i+3] for i in range(0, len(numeric_cols), 3)]      # Creating groups of 3 of columns


        # Create plot for each group
        plot_paths = []
        for i, cols in enumerate(col_groups):
            if get_basic_info(df)['Number of rows'] > 3000 or get_basic_info(df)['Number of columns'] > 10:
                plot_path = plot_group(df, cols, i)
                plot_paths.append(plot_path)
                break

            else:
                plot_path = plot_group(df, cols, i)
                plot_paths.append(plot_path)

        # If there exists only 1 subplot then stopping further execution of this method
        if len(plot_paths) == 1:
            os.rename(plot_paths[0],'distributions.png')
            return 'distributions.png'
        
        final_path = combine_plots(plot_paths)

        return final_path
    
    except Exception as e:
        print(f"Error in distribution_plots: {e}")
        return None

def change_image_encoding(img):
    """Convert images to base64 string for LLM evaluation"""
    try:
        if not os.path.exists(img):
            return None
        with open(img, "rb") as image:
            return base64.b64encode(image.read()).decode('utf-8')
    except Exception as e:
        print(f"Error in change_image_encoding: {e}")
        return None

      

def initial_analyse(csv_file):
    """Perform Initial analysis of the dataset using basic stats, sample values and outliers detected(if any)"""
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

    Provide a comprehensive analysis including:
    1. Data overview
    2. Key patterns and relationships
    3. Notable outliers or anomalies
    4. Potential insights and implications

    A Proper comprehensive theoretical analysis is needed here.
    """
        }
        analysis_response = llm_req([analysis_prompt])          # Calling LLM via requests for response

        if not analysis_response:
            return "Error in generating analysis"
        
        return analysis_response['choices'][0]['message']['content']
    
    except Exception as e:
        print(f"Error in initial_analyse: {e}")
        return "Error in analyzing data"



def generate_python_code(csv_file):
    """
    Generate python code based on basic stats, sample values and outliers detected(if any) in your dataset
    to perform various kind of analysis and create visualizations
    """
    try:
        df = pd.read_csv(csv_file, encoding='unicode_escape')
        
        basic_stats = get_basic_info(df)
        sample_values = get_sample_values(df)
        
        outliers = outlier_detection(df)

        # Prompt for code generation
        code_gen_prompt = {
            "role": "user",
            "content": f"""
            Given the following basic information about the dataset:
            Filename: {csv_file}
            Basic stats: {pd.json_normalize(basic_stats).to_dict(orient='records')[0]}
            Sample values: {json.dumps(sample_values, indent=2)}
            Outlier : {json.dumps(outliers, indent=2)}

            Generate python code that can be directly executed using the exec() function which do analysis that might provide deeper insights into the dataset, such as outlier detection, clustering, regression analysis, time series analysis, or feature importance analysis, also include visualization code in your output and save those charts as *.png but keep the overall code small and limited such that it don't need very high processing power or huge time to run.
            Also do not add any additional content, not even the markdown ```python at the start, just provide pure code content and nothing else in your output
            """}

        code_gen_response = llm_req([code_gen_prompt])          # Calling LLM via requests for response
        
        if not code_gen_response:
            return "Error in generating code"
            
        return code_gen_response['choices'][0]['message']['content']
    
    except Exception as e:
        print(f"Error in generate_python_code: {e}")
        return "Error in generating code"
    

def run_generated_code(response):
    """
    This function tries to run the python code generated by LLM via 'generate_python_code()' method
    """
    # print(response)

    try:
        exec_globals = {}
        exec(response, {}, exec_globals)            # Trying to execute code 
        print("Execution completed successfully.")
        generated_plots = exec_globals.get("generated_plots", [])
    except Exception as e:
        print(f"Error executing LLM-generated code: {e}")
        generated_plots = []

    return generated_plots

def visual_analysis(csv_file):
    """
    Perform visual analysis on the plots and graphs generated by converting them to base64 strings and passing it to LLM for further analysis
    """
    try: 
        df = pd.read_csv(csv_file, encoding='unicode_escape')

        correlation_plot = correlation_heatmap(df)
        distribution_plot = distribution_plots(df)

        # Creating prompt to accomodate the type of visualizations we have and pass it to LLM

        if correlation_plot or distribution_plot:
            vision_prompt = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze these visualizations and provide insights in Markdown format:"}
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
                
            vision_response = llm_req([vision_prompt])          # Calling LLM via requests for response
            vision_analysis = vision_response['choices'][0]['message']['content']
            return vision_analysis
    
    except Exception as e:
        print(f"Error in visual_analysis: {e}")
        return "Error in visual analysis"
    
def main(csv_file):
    """
    Main function where all the steps above are combined to generate a final narration of the story which utilises 
    Function Calling to generate the overview, main story, insights, recommendations based on the prior analysis methods performed !!
    """

    try:
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' does not exist")       # Check whether csv file exists or not
            return

        # Prompt for final narration
        final_prompt = """Combine the following analyses into a cohesive narrative:

        Initial Analysis 
        Visual Analysis
        Generated Code for analysis

        Create a final README.md that:
        1. Narrate a compelling story about the data 
        2. Highlights key insights
        3. Provides actionable recommendations
        4. Properly references the visualizations
        5. Add the generated visualizations with names correlation.png and distributions.png in the file
        6. Include generated code for analysis and how it is suitable 

        Format in Markdown with clear sections and don't add ```markdown in the beginning of your output"""

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
    
    
    

