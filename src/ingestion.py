import os
import pandas as pd
from pathlib import Path


class CSVIngestion:

    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_path)
        except UnicodeDecodeError:
            df = pd.read_csv(self.input_path, encoding="latin1")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def save(self, df: pd.DataFrame) -> None:
        os.makedirs(self.output_path.parent, exist_ok=True)
        df.to_csv(self.output_path, index=False)

    def run(self) -> pd.DataFrame:
        df = self.load()
        df = self.clean(df)
        self.save(df)
        return df


#Script Execution

#replace the paths with actual file paths before running have to be in the scripts where we needed
"""

if __name__ == "__main__":

    ingestion = CSVIngestion(
        input_path="path/to/raw.csv",
        output_path="path/to/interim.csv"
    )
    ingestion.run()


"""