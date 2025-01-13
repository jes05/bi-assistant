import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('D://Mtech//Semester4//bi-assistant-config.ini')

DATA_FILE = config['FILEPATH']['output_path']
main_path = config['FILEPATH']['main_path']
source_path = config['FILEPATH']['source_path']
output_path = config['FILEPATH']['output_path']
user_query_data_path = config['FILEPATH']['user_query_data_path']
save_file_path = config['FILEPATH']['save_file_path']