import pandas as pd
class Preprocess:
    def __init__(self, config: dict):
        self.data = pd.read_csv(config['file_name'], usecols=config['fields'])
    
    def dropna(self):
        self.data = self.data.dropna()
    
    def columnDataType(self, data_type_for_each_column: dict):
        for col in self.data.columns.values:
            self.data[col] = self.data[col].astype(data_type_for_each_column[col])
    
    def write_csv(self, file_name: str):
        self.data.to_csv(f"{file_name}.csv")
    
    def get_data(self) -> pd.DataFrame:
        return self.data
    
