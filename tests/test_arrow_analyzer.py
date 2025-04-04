#!/usr/bin/env python3
import sys
import os
import logging
import traceback
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arrow_analyzer_test")

def main():
    """
    Test script that replicates the Jupyter notebook workflow
    and tests arrow_analyzer.index_arrow_table() function
    """
    # Set RUST_BACKTRACE environment variable
    os.environ["RUST_BACKTRACE"] = "1"
    
    try:
        logger.info("Importing required libraries")
        import polars as pl
        import arrow_analyzer
        
        # Path to the test data
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "test-data", "tmdb.csv")
        logger.info(f"Reading data from {data_path}")
        
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file {data_path} does not exist!")
            return 1
        
        # Read the CSV file
        df = pl.read_csv(data_path)
        logger.info(f"Loaded dataframe with shape: {df.shape}")
        
        # Take just top 100 rows
        #df = df.slice(0, 100)
        #logger.info(f"Reduced to 100 rows: {df.shape}")
        
        # Convert to Arrow
        logger.info("Converting to Arrow table")
        df_arrow = df.to_arrow()
        
        # Call the function that crashes in the notebook
        logger.info("Calling arrow_analyzer.index_arrow_table()")
        
        # Number of retry attempts
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                arrow_analyzer.index_arrow_table(df_arrow)
                logger.info("Successfully executed index_arrow_table")
                return 0
            except Exception as e:
                logger.error(f"Error during index_arrow_table (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Maximum retries reached. Test failed.")
                    logger.error(traceback.format_exc())
                    return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 