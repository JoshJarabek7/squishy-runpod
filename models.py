from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from PIL import Image
import base64
from io import BytesIO

class SegmentationMode(str, Enum):
    """Enum defining the available segmentation modes."""
    SEMANTIC = "semantic"
    AUTOMATIC = "automatic"

class SegmentationInput(BaseModel):
    """Pydantic model for segmentation input parameters."""
    image_base64: str = Field(..., description="Base64 encoded image data")
    text_prompt: Optional[str] = Field(None, description="Text prompt for semantic segmentation")
    automatic: bool = Field(False, description="Whether to use automatic segmentation")

    @validator("image_base64")
    def validate_image(cls, v: str) -> str:
        try:
            # Try to decode and open the image to validate it
            image_data = base64.b64decode(v)
            Image.open(BytesIO(image_data))
            return v
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

    def get_pil_image(self) -> Image.Image:
        """Convert base64 image to PIL Image."""
        image_data = base64.b64decode(self.image_base64)
        return Image.open(BytesIO(image_data)).convert('RGB')

    @property
    def mode(self) -> SegmentationMode:
        """Determine the segmentation mode based on input parameters."""
        return SegmentationMode.SEMANTIC if self.text_prompt else SegmentationMode.AUTOMATIC

    def validate_mode(self) -> None:
        """Validate that only one mode is specified."""
        modes = [bool(self.text_prompt), self.automatic]
        active_modes = sum(modes)

        if active_modes == 0:
            raise ValueError("No segmentation mode specified")
        if active_modes > 1:
            raise ValueError("Multiple segmentation modes specified")

class SegmentationOutput(BaseModel):
    """Pydantic model for segmentation output."""
    image_base64: str = Field(..., description="Base64 encoded segmented image")
    size_bytes: int = Field(..., description="Size of the image in bytes")
    width: int = Field(..., description="Width of the image in pixels")
    height: int = Field(..., description="Height of the image in pixels")
    quality: int = Field(..., description="JPEG quality used (if compressed)")

    @classmethod
    def from_pil_image(cls, image: Image.Image, max_size_bytes: int = 19_000_000) -> "SegmentationOutput":
        """Create SegmentationOutput from PIL Image, ensuring size is under limit."""
        buffer = BytesIO()
        width, height = image.size
        quality = 95
        format = "PNG" if image.mode == "RGBA" else "JPEG"
        
        # First try with high quality
        image.save(buffer, format=format, quality=quality)
        size = buffer.tell()
        
        # If too large, compress with JPEG
        if size > max_size_bytes and format == "PNG":
            # Convert to RGBA on white background
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            
            format = "JPEG"
            buffer = BytesIO()
            image.save(buffer, format=format, quality=quality)
            size = buffer.tell()
        
        # If still too large, reduce quality
        while size > max_size_bytes and quality > 5:
            quality -= 5
            buffer = BytesIO()
            image.save(buffer, format=format, quality=quality)
            size = buffer.tell()
        
        if size > max_size_bytes:
            raise ValueError(f"Unable to compress image below {max_size_bytes} bytes")
        
        return cls(
            image_base64=base64.b64encode(buffer.getvalue()).decode(),
            size_bytes=size,
            width=width,
            height=height,
            quality=quality
        ) 