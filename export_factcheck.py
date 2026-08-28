import os
import csv
import django
import logging

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myaifactcheck.settings')  # Replace 'myaifactcheck' with your project name
django.setup()

from app.models import Factcheck  # Replace 'app' with your app name where the Factcheck model is defined

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def export_factchecks_to_csv():
    # Define the file path for the CSV
    file_path = 'factchecks.csv'
    
    try:
        # Fetch all records from the Factcheck model
        factchecks = Factcheck.objects.all()
        total_records = factchecks.count()
        logger.info(f"Total records fetched: {total_records}")
        
        # Check if there are any records
        if total_records == 0:
            logger.warning("No records found in the Factcheck model.")
            print("No records to export.")
            return

        # Open a CSV file in write mode
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow([
                'ID', 'User Input News', 'Fact Check Result', 'Sentiment Label', 
                'Genuine URLs', 'Non-Authentic URLs', 'Number of Genuine Sources', 
                'Genuine URLs and Dates', 'Non-Authentic Sources', 'Created At'
            ])

            # Write each Factcheck record as a row
            for factcheck in factchecks:
                # Debugging: Log each record being written
                logger.debug({
                    "ID": factcheck.id,
                    "User Input News": factcheck.user_input_news,
                    "Fact Check Result": factcheck.fresult,
                    "Sentiment Label": factcheck.sentiment_label,
                    "Genuine URLs": factcheck.genuine_urls,
                    "Non-Authentic URLs": factcheck.non_authentic_urls,
                    "Number of Genuine Sources": factcheck.num_genuine_sources,
                    "Genuine URLs and Dates": factcheck.genuine_urls_and_dates,
                    "Non-Authentic Sources": factcheck.non_authentic_sources,
                    "Created At": factcheck.created_at,
                })

                # Write to the CSV file
                writer.writerow([
                    factcheck.id,
                    factcheck.user_input_news,
                    factcheck.fresult,
                    factcheck.sentiment_label,
                    factcheck.genuine_urls,
                    factcheck.non_authentic_urls,
                    factcheck.num_genuine_sources,
                    factcheck.genuine_urls_and_dates,
                    factcheck.non_authentic_sources,
                    factcheck.created_at,
                ])

        logger.info(f"Data exported successfully to {file_path}")
        print(f"Data exported successfully to {file_path}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    export_factchecks_to_csv()
