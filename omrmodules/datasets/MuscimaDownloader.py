from omrdatasettools import Downloader, OmrDataset
import os

def download(root="muscima"):
    downloader = Downloader()
    downloader.download_and_extract_dataset(dataset=OmrDataset.MuscimaPlusPlus_V2, 
                                            destination_directory=root)
    downloader.download_and_extract_dataset(dataset=OmrDataset.MuscimaPlusPlus_MeasureAnnotations,
                                            destination_directory=os.path.join(root, "v2.0/data/measure"))
    return root