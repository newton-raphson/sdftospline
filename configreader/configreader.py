import configparser

def parse_config(config_file):
    # Initialize the configparser
    print(f'Parsing config file: {config_file}')
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(config_file)

    # Extract parameters
    mode = config.get('Settings','mode')  # 'train' or 'test'
    batch_size = config.getint('Settings','batch_size')
    learning_rate = config.getfloat('Settings','learning_rate')
    epochs = config.getint('Settings','epochs')
    root_directory = config.get( 'Settings','root_directory')
    save_path = config.get('Settings','save_path')

    return mode, batch_size, learning_rate, epochs, root_directory, save_path

# Example usage:
# config_file_path = 'config1.txt'
# mode, batch_size, learning_rate, epochs, root_directory, save_path = parse_config(config_file_path)

# print(f'Mode: {mode}')
# print(f'Batch Size: {batch_size}')
# print(f'Learning Rate: {learning_rate}')
# print(f'Epochs: {epochs}')
# print(f'Root Directory: {root_directory}')
# print(f'Save Path: {save_path}')
