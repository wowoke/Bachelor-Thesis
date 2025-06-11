
"""TODO."""

# pip install "google-cloud-aiplatform>=1.38"


import logging
logger = logging.getLogger(__name__)

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

def generate_text(project_id: str, location: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-pro-vision")
    # Query the model
    response = multimodal_model.generate_content(
        [
            # Add an example image
            Part.from_uri("gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"),
            # Add an example query
            "what is shown in this image?",
        ]
    )
    return response.text

class GeminiModel:
    def __init__(self, project_id: str, location: str, model_name: str = "gemini-pro-vision"):
        """
        Initialize the Gemini model.
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.multimodal_model = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize Vertex AI and load the model.
        """
        vertexai.init(project=self.project_id, location=self.location)
        self.multimodal_model = GenerativeModel(self.model_name)

    def generate(
        self,
        instruction: str,
        image_path: str = None,
        max_new_tokens: int = 300,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
        stop: str = None,
    ) -> str:
        """
        Generate a response using the Gemini model.

        Args:
            instruction (str): The text instruction or query.
            image_path (str or None): Path to an image file, if provided.
            max_new_tokens (int): Maximum number of tokens in the response.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            repetition_penalty (float): Repetition penalty.
            length_penalty (float): Length penalty.
            num_return_sequences (int): Number of response sequences to generate.
            stop (str or None): Stop string.

        Returns:
            str: The generated response.
        """
        # Prepare the request parts
        request_parts = []

        # If an image path is provided, add it as a part
        if image_path:
            try:
                request_parts.append(Part.from_file(image_path, mime_type="image/jpeg"))
            except Exception as e:
                logger.error(f"Failed to load image from path: {image_path}. Error: {e}")
                raise FileNotFoundError(f"Image file '{image_path}' could not be processed.")

        # Add the text instruction as a part
        request_parts.append(instruction)

        # Generate the response
        try:
            response = self.multimodal_model.generate_content(
                request_parts,
                max_output_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            raise

        # Return the generated response
        return response.text




#  POST https://us-central1-aiplatform.googleapis.com/v1/projects/mygemini-408821/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent

if __name__ == "__main__":
    generate_text("mygemini-408821", "us-central1")

    # Class Example Usage

    project_id = "mygemini-408821"
    location = "us-central1"

    # Initialize the Gemini model
    gemini_model = GeminiModel(project_id=project_id, location=location)

    # Example with text-only input
    response = gemini_model.generate("Explain the process of photosynthesis.")
    print("Text-only response:", response)

    # Example with image and text input
    response_with_image = gemini_model.generate(
        "What is shown in this image?", image_path="path/to/your/image.jpg"
    )
    print("Response with image:", response_with_image)