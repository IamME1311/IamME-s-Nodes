from pathlib import Path
from datetime import datetime
from .py.utils import *

class IamME_Database:
    def __init__(self) -> None:
        self.main_folder_path = Path(r"\\sridhar\Myntra_Backup\Prompt Database")

        if not self.main_folder_path.exists() and not self.main_folder_path.is_dir(): # if main path doesn't exist, go for temporary approach
            self.temp_folder_path = Path(__file__).cwd().joinpath(".temp")
            self.temp_folder_path.mkdir(exist_ok=True) # make temp folder in current folder

            self.temp_db_path = self.temp_folder_path.joinpath("temp_db.json")
            self.temp_db_path.touch(exist_ok=True) # make temp db file
            self.temp_db_path.write_text("[]")

            self.type = "temp"
        else:
            self.main_db_path = self.main_folder_path.joinpath("database.json")
            self.main_db_path.touch(exist_ok=True) # make main db file if it doesn't exist
            self.type = "main"

    def update_db(self, input_prompt:str, output_prompt:str) -> None:
        """ Update the current database with input_prompt and output_prompt"""
        input_prompt = input_prompt.replace("\"", "")
        if self.type == "main":
            with open(self.main_db_path, "r") as f:
                data:list = json.load(f)
            f.close()
            if not data.__class__.__name__ == "list":
                raise TypeError(f"#{PACK_NAME}'s Nodes : database file is not in correct format")

            with open(self.main_db_path, "w") as f:
                data.append({"datetime":datetime.now().strftime("%B %d, %Y %I:%M %p"), "input_prompt":input_prompt, "output_prompt":output_prompt})
                f.write(json.dumps(data)) # save to file
            f.close()

        elif self.type == "temp":
            with open(self.temp_db_path, "r") as f:
                data:list = json.load(f)
            f.close()
            if not data.__class__.__name__ == "list":
                raise TypeError(f"#{PACK_NAME}'s Nodes : database file is not in correct format")

            with open(self.temp_db_path, "w") as f:
                data.append({"datetime":datetime.now().strftime("%B %d, %Y %I:%M %p"), "input_prompt":input_prompt, "output_prompt":output_prompt})
                f.write(json.dumps(data)) # save to file
            f.close()

        log_to_console(f"{self.type} database updated")

    def delete_temp_db(db_path:Path) -> None:
        """ Delete temporary database"""
        if db_path.exists():
            for item in db_path.parent.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)

            db_path.parent.rmdir()
            log_to_console("Successfully removed temp database")
        else:
            log_to_console("Temp database not found!!")

    def merge_DB(self) -> None:
        """Merge temp and main database"""
        temp_db_path = Path(__file__).cwd().joinpath(".temp").joinpath("temp_db.json")
        if temp_db_path.exists() and self.main_folder_path.exists():
            log_to_console("merging databases")
            with open(temp_db_path, "r") as f:
                temp_data = json.load(f)
            f.close()

            with open(self.main_db_path, "r") as m:
                main_data = json.load(m)
            m.close()

            if main_data.__class__.__name__ != "list" or temp_data.__class__.__name__ !="list":
                raise TypeError(f"#{PACK_NAME}'s Nodes : both database datatypes mismatch!!")

            main_data = main_data + temp_data

            with open(self.main_db_path, "w") as f:
                f.write(json.dumps(main_data))
            f.close()

            log_to_console("Databases merged")

            self.delete_temp_db(temp_db_path)
        else:
            log_to_console("No temp DB to merge with, aborting!!")





