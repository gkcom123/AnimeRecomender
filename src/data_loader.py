import pandas as pd
class DataLoader:
    def __init__(self, file_path, processed_path: str = None):
        self.file_path = file_path
        self.processed_path = processed_path

    def load_data(self):
        try:
            data = pd.read_csv(self.file_path, encoding='utf-8', on_bad_lines='skip').dropna()
            required_columns = {'Name' , 'Genres','sypnopsis'}
            missing_columns = required_columns - set(data.columns)
            if missing_columns: 
                raise ValueError(f"Missing required columns: {missing_columns}")
            data["combined_info"] = (
                "Title: " + data["Name"] + 
                " Overview: " + data["sypnopsis"] + 
                " Genres: " + data["Genres"]
            )
            data[['combined_info']].to_csv(self.processed_path, index=False,encoding='utf-8')
            return self.processed_path
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None