from typing import Callable, Optional
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.datasets
from pathlib import Path
from tqdm import tqdm

torchvision.datasets.utils.tqdm = tqdm


class ImageDataset(torchvision.datasets.ImageFolder):
    DOWNLOAD_URL = "https://daiv-cnu.duckdns.org/contest/ai_competition[2025]/dataset/datasets.zip"
    ARCHIVE_FILENAME = "datasets.zip"

    def __init__(
        self,
        root: str,
        force_download: bool = True,
        train: bool = True,
        valid: bool = False,
        unlabeled: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        root = Path(root)
        self.download(root, force=force_download)

        if not train:
            root = root / "test"
        elif valid:
            root = root / "val"
        elif unlabeled:
            root = root / "unlabeled"
        else:
            root = root / "train"

        super().__init__(root=root, transform=transform, target_transform=target_transform)

    @classmethod
    def download(cls, root: Path, force: bool = False):
        zip_path = root / cls.ARCHIVE_FILENAME

        if force or not zip_path.exists():
            download_and_extract_archive(
                cls.DOWNLOAD_URL,
                download_root=root,
                extract_root=root,
                filename=cls.ARCHIVE_FILENAME
            )

            # Arrange Dataset Directory
            for target_dir in [root / "test", root / "unlabeled"]:
                for file_path in target_dir.iterdir():
                    if file_path.is_file() and file_path.suffix == ".JPEG":
                        new_dir = target_dir / file_path.stem
                        new_dir.mkdir(exist_ok=True)
                        file_path.rename(new_dir / file_path.name)

            print("INFO: Dataset archive downloaded and extracted.")
        else:
            print("INFO: Dataset archive found in the root directory. Skipping download.")