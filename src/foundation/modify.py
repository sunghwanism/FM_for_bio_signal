python.exe c:/Users/user/Desktop/Biosignal/FM_for_bio_signal/src/foundation/modify.py
import os

class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def load_data_list(self):
        # Get the list of files/directories in self.data_folder/pair
        data_files = os.listdir(self.data_folder)
        return data_files

# Example usage:
if __name__ == "__main__":
    data_folder = r"C:\Users\user\Downloads\mesa_pair\pair"
    
    # Create an instance of FolderDataLoader
    folder_data_loader = DataLoader(data_folder)
    
    # Load the data list
    data_list = folder_data_loader.load_data_list()
    
    # Print the list of files
    print("List of files in the 'pair' folder:")
    for item in data_list:
        print(item)
    print(len(data_list))
